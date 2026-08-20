#include "ppml/Model.h"
#include "ppml/DataLoader.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"
#include "ppml/PythonBridge.h"
#include "ppml/ONNXExporter.h"
#include "ppml/GradientClipper.h"
#include "ppml/AdamW.h"
#include "ppml/GGUF.h"
#include "ppml/LDDT.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <unistd.h>        // sysconf 用于页大小/物理页数 (内存)
#include <sys/sysinfo.h>   // sysinfo 用于总/可用内存
#include <cuda_runtime.h>  // cudaGetDeviceProperties 用于 GPU 信息

using namespace ppml;

// ============================================================================
// 打印 CPU / 内存 / GPU 信息（训练开始前的环境概览）
// ============================================================================
void print_system_info() {
    // ---- CPU ----
    std::cout << "==== System Info ====" << std::endl;
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    std::cout << "  CPU   : " << ncpu << " logical cores" << std::endl;

    // ---- 内存 (Linux sysinfo) ----
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        const double gb = 1024.0 * 1024.0 * 1024.0;
        std::cout << "  RAM   : total " << std::fixed << std::setprecision(1)
                  << (si.totalram * si.mem_unit / gb) << " GB, "
                  << "available " << (si.freeram * si.mem_unit / gb) << " GB"
                  << std::endl;
    }

    // ---- GPU (CUDA) ----
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0) {
        for (int d = 0; d < dev_count; ++d) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
                std::cout << "  GPU[" << d << "] : " << prop.name
                          << ", compute " << prop.major << "." << prop.minor
                          << ", VRAM " << std::fixed << std::setprecision(2)
                          << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB"
                          << std::endl;
            }
        }
    } else {
        std::cout << "  GPU   : (no CUDA device)" << std::endl;
    }

    // ---- 训练模式开关 ----
    const bool full = (std::getenv("FULL_TRAIN") != nullptr) &&
                      (std::strcmp(std::getenv("FULL_TRAIN"), "true") == 0);
    std::cout << "  Mode  : " << (full ? "FULL_TRAIN" : "dev (FULL_TRAIN unset)")
              << std::endl;
    std::cout << "===========================" << std::endl;
}

// ============================================================================
// 工具: 将值张量 (GraphOutput.coords / ModelInput.true_coords / 各 gt onehot) 包装为图节点 leaf。
// 注意: 训练前向用 PPMLModel::forward_graph（返回可微图节点），loss 的 pred 部分（head logits）
//       直接接图节点，梯度可回传；仅 gt/true 与坐标相关量（coords 为 SE3 图外值更新，非可微）
//       用 wrap_value_as_leaf 断链（坐标梯度暂不接，见 FAPE/conf）。
// ============================================================================
TensorF32* wrap_value_as_leaf(const TensorF32& t, const std::vector<int64_t>& dims) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size() && i < 4; i++) ne[i] = dims[i];
    TensorF32* leaf = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
    // 数据按扁平顺序拷入. Tensor 是 move-only (禁拷贝), 故:
    //   - CUDA 上: 用 t.cpu() 生成新的 CPU 张量 (可移动) 再取 data
    //   - CPU 上:  直接取 t.data() 拷入
    float* dst = bind_leaf_data(context(), leaf);
    if (t.device() == Device::CUDA) {
        TensorF32 tcpu = t.cpu();  // 移动构造, 合法
        std::memcpy(dst, tcpu.data(), tcpu.numel() * sizeof(float));
    } else {
        std::memcpy(dst, t.data(), t.numel() * sizeof(float));
    }
    return leaf;
}

// ============================================================================
// 工具: 将参数 Tensor 的数据安全读取到一块 CPU 内存 (方案 b)
//   - 若 tensor 已迁入 backend buffer, 通过 buffer_->get_tensor 读取
//     (自动处理 CPU/GPU buffer 类型)
//   - 否则若 device 为 CPU, 直接 memcpy data_
//   - CUDA 上无 buffer 的裸张量, 退回 t->cpu()
// 返回: 指向连续 CPU 数据的 buffer (vector 生命期由调用方保证存活到写完)
// ============================================================================
std::vector<float> read_tensor_cpu(const TensorF32* t) {
    size_t bytes = t->nbytes();
    std::vector<float> buf(t->numel());

    if (t->buffer_) {
        // 从 backend buffer 读取 (buffer_->get_tensor 按偏移读)
        // DefaultBuffer get_tensor will deal with CPU/GPU buffer type automatically
        t->buffer_->get_tensor(t, buf.data(), t->buffer_offs_, bytes);
    } else if (t->device() == Device::CPU) {
        std::memcpy(buf.data(), t->data(), bytes);
    } else {
        // CUDA 且无 buffer: 用 t->cpu() 生成一份 CPU 拷贝
        // cuda backend synchronize
        TensorF32 tcpu = t->cpu();   // 移动语义, 临时对象
        std::memcpy(buf.data(), tcpu.data(), bytes);
    }
    return buf;
}

// ============================================================================
// 开发功能: 统计所有权重占用内存大小 (仅计算, 不写文件)
// 返回字节数。同时打印每个张量的形状与字节数 (便于调优量化/显存)。
// ============================================================================
size_t estimate_params_memory(const PPMLModel& model) {
    // params() 收集的指针顺序固定, 但此处通过尺寸计算内存;
    // 由于 params() 为非 const, 这里用一个 const 转换不了, 故通过 const_cast 调用
    // (仅读取 nbytes/shape, 不修改任何状态, 安全)
    auto& m = const_cast<PPMLModel&>(model);
    auto params = m.params();

    size_t total_bytes = 0;
    std::cout << "  [ParamsMemory] #tensors=" << params.size() << std::endl;
    size_t i = 0;
    for (auto* t : params) {
        if (!t || t->numel() == 0) continue;
        size_t bytes = t->nbytes();
        total_bytes += bytes;
        /* if (i < 64) {  // 只打印前 64 个, 避免刷屏
            std::cout << "    tensor_" << i << "  shape=(";
            for (int d = 0; d < t->ndim(); d++) {
                if (d) std::cout << ",";
                std::cout << t->dims()[d];
            }
            std::cout << ")  " << bytes << " B" << std::endl;
        } */
        i++;
    }

    double mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    std::cout << "  [ParamsMemory] total = " << total_bytes << " B = "
              << mb << " MB" << std::endl;
    return total_bytes;
}

// 计算参数总字节数 (供 save_checkpoint 元数据用)
size_t total_bytes_of(const std::vector<TensorF32*>& params) {
    size_t total = 0;
    for (auto* t : params) total += t->nbytes();
    return total;
}

// ============================================================================
// 保存 checkpoint (方案 b): 先把所有参数临时拷贝到 CPU, 再写入 GGUF
//   path: 输出 gguf 文件路径
//   epoch: 当前已完成 epoch (0-based), total_epochs, loss, elapsed_sec
// ============================================================================
void save_checkpoint(PPMLModel& model, const std::string& path,
                     int epoch, int total_epochs, float loss, double elapsed_sec) {
    // 1. 收集参数 (及一一对应的语义名, 含 block/attention 等信息)
    std::vector<TensorF32*> params;
    std::vector<std::string> param_names;
    model.collect_params_with_names(params, param_names);
    if (params.empty()) {
        throw std::runtime_error("save_checkpoint: no parameters to save");
    }
    if (param_names.size() != params.size()) {
        throw std::runtime_error("save_checkpoint: tensor name count mismatch");
    }

    // 2. 方案 b: 临时 CPU 拷贝 — 每张量读成独立 vector, 存活到 save_gguf 完成
    std::vector<std::vector<float>> cpu_copies;
    cpu_copies.reserve(params.size());
    for (auto* p : params) {
        cpu_copies.push_back(read_tensor_cpu(p));
    }

    // 3. 因为 save_gguf 需要 Tensor* 集合, 我们用轻量的非 owning Tensor 视图
    //    指向 CPU 拷贝 (数据连续), 生命周期与 cpu_copies 一致
    std::vector<TensorF32> views;
    views.reserve(params.size());
    std::vector<TensorF32*> gguf_params;
    gguf_params.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        // 注意: Tensor 禁止拷贝/赋值, 但允许移动; 用 emplace 构造非 owning 视图
        //       view(data, shape, CPU, own=false) 不会拥有数据, 不会 free cpu_copies
        views.emplace_back(params[i]->shape(), cpu_copies[i].data(),
                           Device::CPU, /*own=*/false);
        gguf_params.push_back(&views.back());
    }

    // 4. 写入 GGUF, 内嵌训练状态元数据 + 语义化张量名
    save_gguf(gguf_params, path,
              {
                  {"epoch", float(epoch + 1)},       // 1-based epoch 序号
                  {"total_epochs", float(total_epochs)},
                  {"loss", loss},
                  {"elapsed_sec", float(elapsed_sec)},
                  {"param_bytes", float(total_bytes_of(params))},
              },
              {
                  {"arch", "ppml_v1"},
                  {"checkpoint", "train"},
              },
              param_names);
}

int main(int argc, char* argv[]) {
    // 1. 初始化 Python 桥接 (用于数据加载和预处理)
    auto& py = PythonBridge::instance();
    if (!py.initialize()) {
        std::cerr << "Failed to initialize Python" << std::endl;
        return 1;
    }

    // 调试/开发开关：FULL_TRAIN=true 时切换为完整训练配置
    //   - MSA 深度 N 由 128 改回 512
    //   - 开启 checkpoint 权重保存 (ckpt_interval)
    //   - 开启 SE3 训练（forward_graph enable_se3=true，更新坐标）
    const bool full_train =
        (std::getenv("FULL_TRAIN") != nullptr) &&
        (std::strcmp(std::getenv("FULL_TRAIN"), "true") == 0);
    // 调试开关：PPML_DEV_SE3=1 时在 dev 模式（FULL_TRAIN 未设）也开 SE3（enable_se3=true），
    // 但保持小配置（MSA 深度 N=128，非 FULL_TRAIN 的 512），避免全图 OOM，用于验证 SE3 训练链路。
    const bool dev_se3 =
        (std::getenv("PPML_DEV_SE3") != nullptr) &&
        (std::strcmp(std::getenv("PPML_DEV_SE3"), "1") == 0);

    // 训练开始前打印 CPU / 内存 / GPU 环境概览
    print_system_info();

    // 2. 创建模型
    //    block 数固定为开发模式正常值：extra=4, main=8, refine=4。
    PPMLConfig config;
    config.d_msa = 256;
    config.d_pair = 128;
    config.d_state = 32;
    config.n_extra_blocks  = 4;
    config.n_main_blocks   = 8;
    config.n_refine_blocks = 4;
    std::cout << "[config] blocks: extra=" << config.n_extra_blocks
              << " main=" << config.n_main_blocks
              << " refine=" << config.n_refine_blocks << std::endl;
    
    PPMLModel model(config);
    
    // 3. 转移到目标设备。默认 CUDA，但 CUDA 对前向/backward 的众多图 op（repeat/permute/
    //    concat/outer_product 等）未实现，graph_compute 算不出 loss。为验证数值正确性先
    //    用 CPU（所有 op 有 kernel）；CUDA op 补齐后再切回。
    // 默认 CPU 训练；设 PPML_USE_CUDA=1 且显存充足时才尝试 CUDA（scheduler 分配 op）。
    const bool use_cuda =
        (std::getenv("PPML_USE_CUDA") != nullptr) && (std::atoi(std::getenv("PPML_USE_CUDA")) != 0);
    model.to(use_cuda ? Device::CUDA : Device::CPU);
    model.train();
    
    std::cout << "Model created and CPU and CUDA Backend init" << std::endl;
    
    // 4. 加载预训练权重 (通过 Python 桥接)
    // 暂时没有预训练权重, 先注释掉, 待有权重文件后再启用
    // if (argc > 1) {
    //     std::string weights_path = argv[1];
    //     ProteinTools::load_torch_weights(model, weights_path);
    // }
    
    // ============================================================
    // 数据加载: 通过 PPMLDataLoader 从 A3M + CSV mapping + 模板目录加载
    // CSV 提供真实坐标 (true_coords), 作为 FAPE 的 ground truth。
    // csv_path 可以是单个 CSV 文件, 也可以是含多个 *_mapping_results.csv
    // 的目录 (一个 uniprot 序列可能被多个 PDB 结构域覆盖, 自动合并)。
    // template_dir: 模板结构目录 (cif/pdb), 自动过滤与真实值重复的 PDB id。
    // ============================================================
    // 命令行参数: train <a3m> <fasta> <csv_mapping|dir> [template_dir] [hhr]
    // 例如(单 CSV): train query.a3m data/P04637.fasta data/P04637.csv data/P04637_template_coords
    //       或(多 CSV 目录): train query.a3m data/P04637.fasta data/P04637_pdbs data/P04637_template_coords
    std::string a3m_path       = (argc > 1) ? argv[1] : "query.a3m";
    std::string fasta_path     = (argc > 2) ? argv[2] : "data/P04637.fasta";
    std::string csv_path       = (argc > 3) ? argv[3] : "data/P04637_pdbs";  // CSV 文件或目录
    std::string template_dir   = (argc > 4) ? argv[4] : "data/P04637_template_coords";
    std::string hhr_path       = (argc > 5) ? argv[5] : "";  // hhr 放最后, 默认为空

    // 从 FASTA 文件读取查询序列 (只读第一条)
    std::string sequence;
    try {
        sequence = PPMLDataLoader::read_fasta_first_sequence(fasta_path);
        std::cout << "Read query sequence from " << fasta_path
                  << " (L=" << sequence.length() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to read FASTA: " << e.what() << std::endl;
        return 1;
    }

    // MSA 深度 N: 开发模式固定为 128 (缓解 Gallocr 峰值内存)。
    //   注: 此参数决定 MSA 列注意力 softmax [N,N,H,L] 与 FFN 中间件 [d_msa,L,N] 的规模,
    //       N=512 时 softmax 408MB / unary 204MB, 峰值 ~18GB 超 WSL 15GB;
    //       N=128 时 softmax ~25MB / unary ~51MB, 峰值可降到 ~5GB 内跑通训练。
    //   FULL_TRAIN=true 时改回 N=512。
    const int msa_max_seqs = (full_train ? 512 : 128);
    PPMLDataLoader loader("", "", msa_max_seqs, 4, 2048);
    std::cout << "[config] MSA depth N=" << msa_max_seqs
              << " (FULL_TRAIN=" << (full_train ? "true" : "false") << ")" << std::endl;
    ModelInput input = loader.load_from_files(a3m_path, sequence, csv_path, template_dir, hhr_path);
    int L = static_cast<int>(sequence.length());

    std::cout << "Loaded data. L=" << L
              << " true_coords shape=(" << (input.true_coords.numel() > 0 ? 1 : 0)
              << "," << L << ",3,3)"
              << " templates=" << input.template_coords.size() << std::endl;
    for (size_t ti = 0; ti < input.template_coords.size(); ++ti) {
        std::cout << "  template[" << ti << "] id=" << input.template_ids[ti]
                  << " chain=" << input.template_chains[ti]
                  << " residues=" << input.template_residue_counts[ti] << std::endl;
    }

    // ============================================================
    // 打印加载到的所有输入张量形状与大小 (debug 辅助)
    // ============================================================
    auto print_tensor = [](const char* name, const TensorF32& t) {
        std::ostringstream oss;
        oss << "  " << std::setw(18) << std::left << name << " shape=(";
        for (int d = 0; d < t.shape().ndim(); ++d) {
            if (d) oss << ",";
            oss << t.shape().dims[d];
        }
        oss << ") numel=" << t.numel()
            << " bytes=" << (t.nbytes() / 1024.0) << "KB";
        std::cout << oss.str() << std::endl;
    };
    std::cout << "[Data sizes] per-input tensor:" << std::endl;
    print_tensor("msa_latent",   input.msa_latent);
    print_tensor("msa_full",     input.msa_full);
    print_tensor("seq_tokens",   input.seq_tokens);
    print_tensor("t1d",          input.t1d);
    print_tensor("t2d",          input.t2d);
    print_tensor("coords",       input.coords);
    print_tensor("true_coords",  input.true_coords);
    print_tensor("tor_feat",     input.tor_feat);
    print_tensor("template_mask",input.template_mask);
    print_tensor("bond_feats",   input.bond_feats);
    print_tensor("dist_matrix",  input.dist_matrix);
    print_tensor("same_chain",   input.same_chain);
    print_tensor("true_msa",     input.true_msa);
    print_tensor("bert_mask",    input.bert_mask);
    print_tensor("gt_chi",       input.gt_chi);
    print_tensor("chi_mask",     input.chi_mask);
    print_tensor("D_onehot",     input.D_onehot);
    print_tensor("O_onehot",     input.O_onehot);
    print_tensor("T_onehot",     input.T_onehot);
    print_tensor("P_onehot",     input.P_onehot);
    print_tensor("pair_mask",    input.pair_mask);
    print_tensor("ca_mask",      input.ca_mask);
    std::cout << "  [Data sizes] total input memory="
              << (input.msa_latent.nbytes() + input.msa_full.nbytes()
                  + input.seq_tokens.nbytes() + input.t1d.nbytes()
                  + input.t2d.nbytes() + input.coords.nbytes()
                  + input.true_coords.nbytes() + input.tor_feat.nbytes()
                  + input.template_mask.nbytes() + input.bond_feats.nbytes()
                  + input.dist_matrix.nbytes() + input.same_chain.nbytes()
                  + input.true_msa.nbytes() + input.bert_mask.nbytes()
                  + input.gt_chi.nbytes() + input.chi_mask.nbytes()
                  + input.D_onehot.nbytes() + input.O_onehot.nbytes()
                  + input.T_onehot.nbytes() + input.P_onehot.nbytes()
                  + input.pair_mask.nbytes() + input.ca_mask.nbytes()) / 1024.0
              << " KB" << std::endl;

    // 若没有真实坐标, 用一个占位 coords 供前向使用
    if (input.coords.numel() == 0) {
        input.coords = zeros<float>({1, L, 3, 3}, Device::CPU);
    }
    if (input.true_coords.numel() == 0) {
        input.true_coords = zeros<float>({1, L, 3, 3}, Device::CPU);
    }
    // 模型在 CUDA 上, 把输入搬到 CUDA
    //input.msa_latent = input.msa_latent.to(Device::CUDA);
    //input.seq_tokens = input.seq_tokens.to(Device::CUDA);
    //input.coords     = input.coords.to(Device::CUDA);
    //input.true_coords= input.true_coords.to(Device::CUDA);
    
    // 5. 训练循环
    // 默认 10 epoch；可用 PPML_NUM_EPOCHS 覆盖（如快速验证 grad_norm 用 3~5）。
    int num_epochs = 10;
    if (const char* pe = std::getenv("PPML_NUM_EPOCHS")) {
        int v = std::atoi(pe);
        if (v >= 1) num_epochs = v;
    }

    // ---- Checkpoint 配置 ----
    // 每隔 checkpoint_interval 个 epoch 保存一次权重 (GGUF)。
    // FULL_TRAIN=true 开启 checkpoint 保存（每 2 epoch）；否则 dev 模式默认不保存。
    int ckpt_interval = full_train ? 2 : 0;    // 0 = 禁用 checkpoint
    const std::string ckpt_dir = "checkpoints"; // checkpoint 输出目录

    // ---- AdamW 优化器配置 (decoupled weight decay) ----
    const float learning_rate   = 1e-4f;   // 峰值学习率
    const float weight_decay    = 0.01f;   // 权重衰减系数（仅对 Linear/Embedding 的 weight 生效）
    AdamW optimizer(learning_rate, weight_decay);
    bool  optimizer_inited = false;

    // 可选: 开发期统计所有权重的内存占用 (不写文件)
    // 若要打印初始权重占用的内存, 取消下面一行注释:
    // size_t init_param_bytes = estimate_params_memory(model);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ===== 计时代码: epoch 级 + 前向/损失阶段子计时 =====
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // 参数膨胀诊断（PPML_DEBUG_PARAM_DIST=1）：每个 epoch 前打印所有参数 max_abs 分布，
        // 用于区分"训练中权重/gamma 膨胀（梯度爆炸）" vs "前向 kernel 写坏参数"。
        //   - 若 max_abs 随 epoch 增长到 >>Xavier 尺度（64 维应 ~0.18，32 维 ~0.25）→ 训练不稳定；
        //   - 若第 1 epoch 前就已巨大 → 初始化/加载/构造 bug 或前向写坏。
        if (getenv("PPML_DEBUG_PARAM_DIST")) {
            std::vector<TensorF32*> pvec;
            std::vector<std::string> pnames;
            model.collect_params_with_names(pvec, pnames);
            int cnt = 0;
            double gmax = 0; int gidx = -1; double gsum = 0;
            int n_null = 0;   // data()==nullptr 的参数数（若>0 → 参数创建后没数据，transfer 也跳过）
            for (size_t i = 0; i < pvec.size(); i++) {
                TensorF32* t = pvec[i];
                if (!t->data()) { n_null++; continue; }
                float mx = 0;
                for (int64_t q = 0; q < t->numel(); q++) {
                    float v = t->data()[q];
                    if (v != v) { mx = mx; continue; }
                    float a = (v < 0) ? -v : v;
                    if (a > mx) mx = a;
                }
                if (mx > 1e6f) cnt++;
                gsum += (double)mx;
                if (mx > gmax) { gmax = mx; gidx = (int)i; }
            }
            fprintf(stderr, "[param-dist] epoch=%d n_params=%zu n_null=%d n_overflow(>1e6)=%d gmax=%.6g@[%d] gsum=%.6g",
                    epoch, pvec.size(), n_null, cnt, gmax, gidx, gsum);
            // 打印 gmax 对应参数的地址与是否被 gallocr 绑定（对比 GRAPH_DEBUG_KERNEL 里 src1 ptr，
            // 判断 kernel 读的"权重"是不是这个真实参数：若 param 正常(±1)而 kernel src1 读巨大/非负，
            // 且二者 ptr 不同 → kernel 读的是被复用的 buffer 而非参数）。
            if (gidx >= 0 && gidx < (int)pvec.size()) {
                TensorF32* gtp = pvec[gidx];
                fprintf(stderr, " ptr=%p buf=%d", (const void*)gtp->data(), (gtp->buffer_ ? 1 : 0));
            }
            fprintf(stderr, "\n");
        }
        
        float epoch_loss = 0.0f;
        
        // 前向传播（图模式：返回可微图节点，供 loss 组装计算图）
        auto fwd_start = std::chrono::high_resolution_clock::now();
        // enable_se3：FULL_TRAIN=true 或 PPML_DEV_SE3=1 时开启 SE3 3D track（训练更新坐标）；
        // dev 默认关闭（run_se3_structural 曾为未完成崩溃，用于小样本流程验证）。
        auto go = model.forward_graph(input, /*enable_se3=*/(full_train || dev_se3));
        auto fwd_end = std::chrono::high_resolution_clock::now();
        auto fwd_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fwd_end - fwd_start).count();
        
        // ============================================================
        // 组装 total_loss 的 5 个组成部分 (调用正确的损失函数)
        //   total_loss(loss_fape, loss_chi, loss_distogram, loss_msa, loss_conf)
        // ============================================================
        const int B = 1;
        const int N_atoms  = B * L * 3;   // 每残基 N/CA/C
        const int N_frames = B * L;

        // --- pred / true coords 展平为 (N_atoms, 3) 图节点 ---
        // 注: coords 为 SE3 图外值更新（非可微），FAPE 对 coords 的反向暂不接（沿用原值版方式，
        //     wrap_value_as_leaf 断链仅影响坐标相关梯度，不影响 msa/pair/state/head 图节点）。
        TensorF32 pred_flat = go.coords.view({N_atoms, 3});
        TensorF32 true_flat = input.true_coords.view({N_atoms, 3});
        TensorF32* pred_node = wrap_value_as_leaf(pred_flat, {N_atoms, 3});
        TensorF32* true_node = wrap_value_as_leaf(true_flat, {N_atoms, 3});

        // --- FAPE frame indices: 每残基一帧 [N, CA, C] ---
        TensorF32* frame_idx = build_frame_atom_indices(B, L);          // (B*L, 3)
        TensorF32* frames_mask    = constant_ones({N_frames});           // (B*L,)
        TensorF32* positions_mask = constant_ones({N_atoms});            // (B*L*3,)

        // 1) FAPE loss
        FAPEConfig fape_cfg;
        fape_cfg.length_scale = 10.0f;
        fape_cfg.d_clamp      = 10.0f;
        fape_cfg.epsilon      = 1e-4f;
        TensorF32* loss_fape_node =
            loss(fape_loss(pred_node, true_node, frame_idx, frames_mask, positions_mask, fape_cfg));

        // 2) Chi (扭转角) loss — 接入 supervised_chi_loss(unnormed, gt, chi_mask, seq_mask)
        //    unnormed : go.alpha 图节点 [14,L,B] → view {2,7,N}（N=B*L，14=7*2，sin/cos 最内）
        //    gt       : input.gt_chi  (B,L,7,2) → view (B*L,7,2) → wrap 图 {2,7,N}
        //    chi_mask : input.chi_mask(B,L,7)   → view (B*L,7)   → wrap 图 {7,N}
        //    seq_mask : 全 1 (B*L,1) → wrap 图 {1,N}
        TensorF32* loss_chi_node = loss(constant_scalar(0.0f));
        if (input.gt_chi.numel() > 0 && go.alpha != nullptr) {
            int64_t chi_N = input.gt_chi.shape().dims[0] * input.gt_chi.shape().dims[1]; // B*L
            TensorF32 chi_gt       = input.gt_chi.view({chi_N, 7, 2});
            TensorF32 chi_mask_2d  = input.chi_mask.view({chi_N, 7});
            TensorF32 seq_mask_2d  = TensorF32({chi_N, 1}, Device::CPU);
            seq_mask_2d.zero_();
            for (int64_t i = 0; i < chi_N; ++i) seq_mask_2d.data()[i] = 1.0f;  // 全残基有效
            TensorF32* unnormed_node = view(go.alpha, Shape{2, 7, chi_N});  // 图节点（梯度回传）
            TensorF32* gt_node       = wrap_value_as_leaf(chi_gt,       {2, 7, chi_N});
            TensorF32* cmask_node    = wrap_value_as_leaf(chi_mask_2d,  {7, chi_N});
            TensorF32* smask_node    = wrap_value_as_leaf(seq_mask_2d,  {1, chi_N});
            loss_chi_node = loss(supervised_chi_loss(unnormed_node, gt_node, cmask_node, smask_node, 0.5f, 0.5f));
        }

        // 3) Distogram loss — 接入 distogram_loss(4 logits, 4 onehot, pair_mask)
        //    布局: logits 用 go.* 图节点 view 为 {bins, B*L*L}（bins 最内, B=1）
        //          gt onehot 与 pair_mask 仍用 leaf
        //    distogram_loss 内 sum_rows 沿 dims[0]=bins(最内) 求和 → per-pair CE
        TensorF32* loss_distogram_node = loss(constant_scalar(0.0f));
        if (input.D_onehot.numel() > 0 && go.distogram != nullptr) {
            int64_t dg_N = input.D_onehot.shape().dims[0]
                         * input.D_onehot.shape().dims[1]
                         * input.D_onehot.shape().dims[2];  // B*L*L
            TensorF32 dg_D    = input.D_onehot.view({dg_N, 60});
            TensorF32 dg_O    = input.O_onehot.view({dg_N, 36});
            TensorF32 dg_T    = input.T_onehot.view({dg_N, 36});
            TensorF32 dg_P    = input.P_onehot.view({dg_N, 18});
            TensorF32 dg_mask = input.pair_mask.view({dg_N});
            TensorF32* l_dist = view(go.distogram, Shape{60, dg_N});   // 图节点
            TensorF32* l_omg  = view(go.omega,     Shape{36, dg_N});
            TensorF32* l_tht  = view(go.theta,     Shape{36, dg_N});
            TensorF32* l_phi  = view(go.phi,       Shape{18, dg_N});
            TensorF32* l_D    = wrap_value_as_leaf(dg_D,    {60, dg_N});
            TensorF32* l_O    = wrap_value_as_leaf(dg_O,    {36, dg_N});
            TensorF32* l_T    = wrap_value_as_leaf(dg_T,    {36, dg_N});
            TensorF32* l_P    = wrap_value_as_leaf(dg_P,    {18, dg_N});
            TensorF32* l_pm   = wrap_value_as_leaf(dg_mask, {dg_N});
            loss_distogram_node = loss(distogram_loss(l_dist, l_omg, l_tht, l_phi,
                                                       l_D, l_O, l_T, l_P, l_pm));
        }

        // 4) Masked MSA loss — 接入 masked_msa_loss(logits, true_msa, bert_mask)
        //    logits   : go.msa_logits 图节点 [23,L,N,B] → view {23, L, N} (dims[0]=类别最内)
        //    true_msa : input.true_msa   (B=1, N, L)    → view (N, L) → 图 {L, N}
        //    bert_mask: input.bert_mask  (B=1, N, L)    → view (N, L) → 图 {L, N}
        //    布局: masked_msa_loss 期望 logits[N_seq,N_res,23], true_msa[N_seq,N_res]
        //          (ggml: logits dims={23,L,N}, true_msa dims={L,N} → N_res=L, N_seq=N)
        TensorF32* loss_msa_node = loss(constant_scalar(0.0f));
        if (input.true_msa.numel() > 0 && go.msa_logits != nullptr) {
            int N_seq_msa = static_cast<int>(input.true_msa.shape().dims[1]);  // N_seq
            TensorF32 msa_true_2d   = input.true_msa.view({N_seq_msa, L});         // (N,L)
            TensorF32 msa_mask_2d   = input.bert_mask.view({N_seq_msa, L});        // (N,L)
            TensorF32* logits_node = view(go.msa_logits, Shape{23, L, N_seq_msa}); // 图节点
            TensorF32* true_node   = wrap_value_as_leaf(msa_true_2d,   {L, N_seq_msa});
            TensorF32* mask_node   = wrap_value_as_leaf(msa_mask_2d,   {L, N_seq_msa});
            loss_msa_node = loss(masked_msa_loss(logits_node, true_node, mask_node));
        }

        // 5) Confidence (pLDDT) loss — 接入 plddt_loss(logits, lddt_onehot, ca_mask)
        //    logits    : go.lddt 图节点 [50,L,B] → view {50, B*L}
        //    ca_mask   : input.ca_mask(B,L)   → view (B*L)     → 图 {B*L}
        //    lddt_onehot: 用 go.coords(预测, 值) vs input.true_coords(真实) 动态计算
        //                 (compute_lddt_ca + lddt_to_onehot), → (B*L,50) → 图 {50, B*L}
        TensorF32* loss_conf_node = loss(constant_scalar(0.0f));
        if (input.ca_mask.numel() > 0 && go.lddt != nullptr && go.coords.numel() > 0) {
            int64_t pl_N = go.coords.shape().dims[0] * go.coords.shape().dims[1];  // B*L
            // 提取 CA 坐标 (B,L,3,3)→(B*L,3), 真值 CA (先转 CPU 以便 .data() 访问)
            TensorF32 pred_coords_cpu = go.coords.cpu();
            TensorF32 true_coords_cpu = input.true_coords.cpu();
            TensorF32 ca_mask_cpu     = input.ca_mask.cpu();
            std::vector<float> pred_ca(pl_N * 3, 0.0f), true_ca(pl_N * 3, 0.0f);
            std::vector<float> ca_m(pl_N, 0.0f);
            for (int64_t i = 0; i < pl_N; ++i) {
                // CA 原子索引=1
                pred_ca[i*3+0] = pred_coords_cpu.data()[i*9 + 1*3 + 0];
                pred_ca[i*3+1] = pred_coords_cpu.data()[i*9 + 1*3 + 1];
                pred_ca[i*3+2] = pred_coords_cpu.data()[i*9 + 1*3 + 2];
                true_ca[i*3+0] = true_coords_cpu.data()[i*9 + 1*3 + 0];
                true_ca[i*3+1] = true_coords_cpu.data()[i*9 + 1*3 + 1];
                true_ca[i*3+2] = true_coords_cpu.data()[i*9 + 1*3 + 2];
                ca_m[i] = ca_mask_cpu.data()[i];
            }
            const int N_BINS = 50;
            std::vector<float> lddt(pl_N, 0.0f), onehot(pl_N * N_BINS, 0.0f);
            compute_lddt_ca(pred_ca.data(), true_ca.data(), ca_m.data(),
                            static_cast<int>(pl_N), 15.0f, lddt.data());
            lddt_to_onehot(lddt.data(), static_cast<int>(pl_N), N_BINS, onehot.data());

            TensorF32 pl_onehot(Shape({pl_N, N_BINS}), onehot.data(), Device::CPU, false);
            TensorF32 pl_camask = input.ca_mask.view({pl_N});
            TensorF32* l_lddt  = view(go.lddt, Shape{N_BINS, pl_N});   // 图节点
            TensorF32* l_oh    = wrap_value_as_leaf(pl_onehot, {N_BINS, pl_N});
            TensorF32* l_cam   = wrap_value_as_leaf(pl_camask, {pl_N});
            loss_conf_node = loss(plddt_loss(l_lddt, l_oh, l_cam));
        }

        // --- total_loss: 0.5*FAPE + 0.5*Chi + 0.3*Distogram + 2.0*MSA + 0.01*Conf ---
        TensorF32* total_node = loss(total_loss(
            loss_fape_node, loss_chi_node, loss_distogram_node, loss_msa_node, loss_conf_node));

        // ============================================================
        // 构建 backward 计算图并执行 (全局裁剪)
        // ============================================================
        PPMLContext* ctx = &context();
        ComputeGraph* cgraph = ComputeGraph::new_graph(ctx);
        cgraph->build_forward_expand(total_node);
        // 开发诊断：打印 total 链关键节点（loss 分量及消费它的 scale/add）的 index，验证拓扑顺序
        if (getenv("GRAPH_DEBUG_LOSS")) {
            TensorF32* loss_nodes[5] = { loss_fape_node, loss_chi_node, loss_distogram_node, loss_msa_node, loss_conf_node };
            const char* lnames[5] = { "fape", "chi", "dist", "msa", "conf" };
            for (int i = 0; i < cgraph->n_nodes(); i++) {
                TensorF32* nn = cgraph->graph_node(i);
                // 打印 loss 分量 index 和引用它们的 scale/add 的 index
                for (int k = 0; k < 5; k++) {
                    if (nn == loss_nodes[k]) {
                        std::cout << "  [order] idx=" << i << " is_loss=" << lnames[k] << " op=" << nn->op << std::endl;
                    }
                }
                if (nn->op == 33 || nn->op == 2) {
                    for (int s = 0; s < 4; s++) {
                        for (int k = 0; k < 5; k++) {
                            if (nn->src[s] == loss_nodes[k]) {
                                std::cout << "  [order] idx=" << i << " consumes_loss=" << lnames[k]
                                          << " op=" << nn->op << std::endl;
                            }
                        }
                    }
                }
            }
        }
        // 开发开关：PPML_NO_BACKWARD=1 跳过反向，仅看前向 loss（隔离 forward/backward nan 来源）
        static const bool no_backward =
            (std::getenv("PPML_NO_BACKWARD") != nullptr) && (std::atoi(std::getenv("PPML_NO_BACKWARD")) != 0);
        if (!no_backward) {
            cgraph->build_backward_expand(ctx, nullptr);

            // 关键：为 loss 节点梯度种子 = 1.0（dL/dL=1）。
            // build_backward_expand 只创建 loss 的梯度累加器（初值 0），不置 1；
            // 若不置 1，反向从 loss 处梯度恒 0 → 所有参数梯度恒 0（grad_norm=0）。
            if (total_node) {
                // loss 梯度累加器是 ctx->new_tensor（no_alloc 下 data==nullptr）。
                // 分配 host 存储并置种子 1.0（loss 为标量 numel==1，直接 p[0]=1）。
                TensorF32* loss_grad = cgraph->graph_get_grad(total_node);
                if (loss_grad && loss_grad->data() == nullptr && loss_grad->numel() == 1) {
                    float* p = bind_leaf_data(*ctx, loss_grad);
                    p[0] = 1.0f;
                }
            }
        }

        // ---- 反向计算 + 梯度裁剪 + AdamW 参数更新 ----
        // 诊断：graph_compute 前检查参数是否已含 NaN（区分"初始化 NaN"vs"graph_compute 后 buffer 覆盖假象"）
        if (getenv("PPML_PRE_COMPUTE_PARAM")) {
            std::cout << "[pre-param] checking params BEFORE graph_compute:" << std::endl;
            for (int gi = 0; gi < cgraph->n_nodes(); ++gi) {
                TensorF32* nd = cgraph->graph_node(gi);
                if (!nd || !(nd->flag & TENSOR_FLAG_PARAM)) continue;
                const float* d = nd->data();
                if (!d) { std::cout << "  [pre-param] node=" << gi << " numel=" << nd->numel() << " data=NULL\n"; continue; }
                bool nan = false; float mn=1e30f, mx=-1e30f;
                for (int64_t q=0; q<nd->numel(); ++q){ float v=d[q]; if(v!=v){nan=true;break;} mn=std::min(mn,v); mx=std::max(mx,v);}
                std::cout << "  [pre-param] node=" << gi << " op=" << nd->op
                          << " numel=" << nd->numel() << " has_nan=" << nan
                          << " min=" << mn << " max=" << mx << std::endl;
            }
        }
        // CUDA 激活时用 scheduler 分配算子（受支持 op 跑 GPU、不支持的跨后端回落 CPU）；
        // 否则走单后端（CPU 或 CUDA）graph_compute。
        Backend* backend = model.active_backend();
        Status compute_st = Status::SUCCESS;
        BackendScheduler* sched = model.scheduler();
        // 仅在显式 PPML_CUDA_SCHED=1 时用 scheduler 分配算子（实验性）。
        // 注意：scheduler 会把参数/梯度放 device，clip_grad_norm/AdamW 当前仍读 host
        //      grad->data()，故全 CUDA 图训练须先补齐 device→host 梯度读回（见说明）。
        static const bool sched_flag =
            (std::getenv("PPML_CUDA_SCHED") != nullptr) && (std::atoi(std::getenv("PPML_CUDA_SCHED")) != 0);
        const bool use_sched = (model.device() == Device::CUDA && sched && sched_flag);
        if (use_sched) {
            sched->split_graph(cgraph);
            if (sched->alloc_splits()) {
                compute_st = sched->graph_compute();   // 跨后端拷贝 + 各 split 执行
            } else {
                compute_st = Status::ALLOC_FAILED;
            }
            if (compute_st == Status::ALLOC_FAILED) {
                // 显存不足/分配失败 → 回退到单后端 CPU 全图计算（保证训练不中断）
                std::cerr << "[WARN] CUDA scheduler alloc failed; "
                          << "falling back to CPU single-backend compute." << std::endl;
                compute_st = backend->graph_compute(cgraph);
            }
        } else {
            compute_st = backend->graph_compute(cgraph); // 执行前向+反向，写入参数梯度
        }
        if (compute_st != Status::SUCCESS) {
            std::cout << "[WARN] graph_compute status=" << static_cast<int>(compute_st)
                      << " (0=SUCCESS 1=ALLOC_FAILED)" << std::endl;
        }

        // ---- 开发诊断：pred_coords 值 + 是否被 gallocr buffer 复用覆盖 ----
        if (getenv("GRAPH_DEBUG_GALLOCR") && pred_node) {
            const int64_t pdN = pred_node->numel();
            const int64_t nprint = (pdN < 9) ? pdN : 9;
            float* pd = pred_node->data();
            std::cout << "[gallocr][pred_coords] numel=" << pdN
                      << " data=" << (void*)pd << std::endl;
            if (pd) {
                bool has_nan = false; float mn = 1e30f, mx = -1e30f;
                for (int64_t q = 0; q < pdN; ++q) {
                    if (pd[q] != pd[q]) { has_nan = true; break; }
                    mn = std::min(mn, pd[q]); mx = std::max(mx, pd[q]);
                }
                std::cout << "  [gallocr][pred_coords] head=[";
                for (int64_t q = 0; q < nprint; ++q)
                    std::cout << pd[q] << (q + 1 < nprint ? "," : "");
                std::cout << "] min=" << mn << " max=" << mx
                          << " has_nan=" << has_nan << std::endl;
            }
            // 找与 pred_node 分配区间重叠的 gallocr managed 节点（覆盖源）
            std::cout << "  [gallocr][pred_coords] aliasing pred_node:" << std::endl;
            backend->gallocr().diagnose_aliasing(pred_node);
            std::cout << "  [gallocr][pred_coords] aliasing true_node:" << std::endl;
            backend->gallocr().diagnose_aliasing(true_node);
        }
        // ---- 开发诊断：distogram head logits 是否含 nan（数据已算好）----
        if (getenv("GRAPH_DEBUG_GALLOCR") && go.distogram && go.distogram->numel() > 0) {
            auto dump4 = [&](TensorF32* t, const char* tag) {
                if (!t || !t->data()) { std::cout << "[dist_logits] " << tag << " data=null\n"; return; }
                const float* gd = t->data();
                float mn=1e30f, mx=-1e30f; bool nan=false; long nnan=0;
                for (int64_t q=0; q<t->numel(); ++q){ float v=gd[q]; if(v!=v){nan=true;++nnan;} mn=std::min(mn,v); mx=std::max(mx,v);}
                std::cout << "[dist_logits] " << tag << " numel=" << t->numel() << " min="<<mn<<" max="<<mx
                          << " nan="<<nan<<" nnan="<<nnan<<" head0="<<gd[0]<<" head1="<<gd[1]<<"\n";
            };
            dump4(go.distogram, "dist");
            dump4(go.omega,     "omega");
            dump4(go.theta,     "theta");
            dump4(go.phi,       "phi");
            // 检查 distogram one-hot 标签本身是否含 NaN（根因验证）
            auto dumpOH = [&](const TensorF32& t, const char* tag){
                const float* d=t.data(); bool nan=false; long nn=0;
                for(int64_t q=0;q<t.numel();++q){ if(d[q]!=d[q]){nan=true;++nn;} }
                std::cout << "[onehot] " << tag << " numel=" << t.numel() << " nan="<<nan<<" nnan="<<nn<<"\n";
            };
            dumpOH(input.D_onehot, "D");
            dumpOH(input.O_onehot, "O");
            dumpOH(input.T_onehot, "T");
            dumpOH(input.P_onehot, "P");
            if (go.lddt && go.lddt->numel()>0) dump4(go.lddt, "lddt");
            if (go.pair && go.pair->numel()>0) dump4(go.pair, "pair");
            if (go.state && go.state->numel()>0) dump4(go.state, "state");
            // 打印 omega/theta/state 中 nan 的确切下标（首 20 个）
            {
                auto nanidx = [&](TensorF32* t, const char* tag){
                    if(!t||!t->data()) return; const float* d=t->data(); int cnt=0;
                    std::cout << "[nanidx] " << tag << " ndim=" << t->shape().ndim();
                    for(int i=0;i<t->shape().ndim();++i) std::cout<<" d"<<i<<"="<<t->shape().dims[i];
                    std::cout << " nan@";
                    for(int64_t q=0;q<t->numel()&&cnt<20;++q){ if(d[q]!=d[q]){ std::cout<<q<<","; ++cnt; } }
                    std::cout<<" cnt="<<cnt<<"\n";
                };
                nanidx(go.omega, "omega");
                nanidx(go.theta, "theta");
                nanidx(go.state, "state");
            }
        }
        // 诊断：统计图节点数与参数梯度幅值（确认 backward 是否产生非零梯度）
        if (getenv("PPML_DEBUG_GRAD")) {
            double gsum = 0, gmx = 0; int gcnt = 0; double gnan = 0;
            // 逐参数梯度统计：定位爆炸源（哪个参数贡献了 max_abs / 大部分 L2）
            struct PStat { double l2=0; double maxabs=0; int64_t numel=0; int idx=0; };
            std::vector<PStat> pstats;
            for (int gi = 0; gi < cgraph->n_nodes(); ++gi) {
                TensorF32* nd = cgraph->graph_node(gi);
                if (!(nd->flag & TENSOR_FLAG_PARAM)) continue;
                TensorF32* gr = cgraph->graph_get_grad(nd);
                if (!gr) continue;
                ++gcnt;
                std::vector<float> gv = read_tensor_cpu(gr);
                PStat ps; ps.numel = gr->numel(); ps.idx = gi;
                for (float v : gv) {
                    if (v != v) { gnan += 1.0; continue; }
                    ps.l2 += (double)(v*v);
                    ps.maxabs = std::max(ps.maxabs, (double)std::fabs(v));
                }
                pstats.push_back(ps);
            }
            for (const auto& ps : pstats) { gsum += ps.l2; gmx = std::max(gmx, ps.maxabs); }
            std::cout << "[grad] n_nodes=" << cgraph->n_nodes()
                      << " params_with_grad=" << gcnt
                      << " sum_sq=" << gsum
                      << " max_abs=" << gmx
                      << " nan_cnt=" << gnan << std::endl;
            // 按 L2 贡献排序，打印 top-N 参数（每参数 L2、max_abs、numel、图节点 idx）
            std::vector<const PStat*> ord;
            for (const auto& ps : pstats) ord.push_back(&ps);
            std::sort(ord.begin(), ord.end(),
                [](const PStat* a, const PStat* b){ return a->l2 > b->l2; });
            int nprint = (int)ord.size() < 12 ? (int)ord.size() : 12;
            for (int q = 0; q < nprint; q++) {
                std::cout << "  [grad-param] rank=" << q
                          << " node_idx=" << ord[q]->idx
                          << " numel=" << ord[q]->numel
                          << " l2=" << ord[q]->l2
                          << " max_abs=" << ord[q]->maxabs << std::endl;
            }
        }
        // 定位纯 forward 里第一个产生 NaN/巨大值(>1e6) 的图节点（hash 修复后完整图执行，用于定位 NaN op 源）
        if (getenv("PPML_DEBUG_NANOP")) {
            int64_t hit_cnt = 0;
            for (int gi = 0; gi < cgraph->n_nodes(); ++gi) {
                TensorF32* nd = cgraph->graph_node(gi);
                if (!nd) continue;
                TensorF32* ndv = nd->data() ? nd : nullptr;
                if (!ndv) continue;
                const int64_t nelt = nd->numel();
                if (nelt <= 0) continue;
                std::vector<float> vals = read_tensor_cpu(nd);
                bool bad = false; float firstbad = 0; int64_t firstidx = -1;
                for (int64_t v = 0; v < (int64_t)vals.size(); ++v) {
                    float x = vals[v];
                    if (x != x || std::fabs(x) > 1e6f) { bad = true; firstbad = x; firstidx = v; break; }
                }
                if (bad) {
                    std::cout << "[nanop] node_idx=" << gi << " op=" << nd->op
                              << " numel=" << nelt << " ndim=" << nd->shape().ndim()
                              << " bad=" << firstbad << " @flat=" << firstidx
                              << " src0_op=" << (nd->src[0] ? (int)nd->src[0]->op : -1)
                              << " src1_op=" << (nd->src[1] ? (int)nd->src[1]->op : -1);
                    if (nd->shape().ndim() >= 1 && nd->shape().ndim() <= 4) {
                        std::cout << " dims=[";
                        for (int d = 0; d < nd->shape().ndim(); ++d)
                            std::cout << (d ? "," : "") << nd->shape().dims[d];
                        std::cout << "]";
                    }
                    std::cout << std::endl;
                    if (++hit_cnt >= 20) break;
                }
            }
            std::cout << "[nanop] total_bad_nodes_shown=" << hit_cnt << std::endl;
        }
        // 全局梯度裁剪阈值：默认 0.1（AF2 惯例），可用环境变量 PPML_CLIP_NORM 覆盖。
        float grad_norm = 0.0f;
        {
            float clip_norm = 0.1f;
            if (const char* cn = getenv("PPML_CLIP_NORM")) {
                float v = static_cast<float>(std::atof(cn));
                if (v > 0.0f) clip_norm = v;
            }
            grad_norm = clip_grad_norm(cgraph, clip_norm);
        }

        if (!optimizer_inited) {
            optimizer.init_from_graph(cgraph);             // 首次收集参数并分配 m/v
            optimizer_inited = true;
            std::cout << "[AdamW] initialized, params=" << optimizer.param_count()
                      << std::endl;
        }
        optimizer.step(cgraph);                            // 更新权重 (decoupled weight decay)

        // 读取 total loss 数值 (graph_compute 后才有值)
        // 用 read_tensor_cpu 走 buffer_->get_tensor，CUDA 上也会同步 D2H，保证读到 host 标量。
        float batch_loss = 0.0f;
        if (total_node->numel() == 1 && (total_node->data() != nullptr || total_node->buffer_)) {
            std::vector<float> tv = read_tensor_cpu(total_node);
            if (!tv.empty()) batch_loss = tv[0];
        }

        // 开发诊断：打印各损失分量（定位 loss=0 根因）
        if (getenv("GRAPH_DEBUG_LOSS")) {
            auto print_loss = [](const char* name, TensorF32* n) {
                if (n && n->data() && n->numel() == 1)
                    std::cout << "  [loss] " << name << " = " << n->data()[0] << std::endl;
                else if (n && n->data()) {
                    // 非标量 loss：打印完整 sum 判断是否含 nan
                    double s = 0; int64_t cn = n->numel();
                    for (int64_t q = 0; q < cn; q++) s += n->data()[q];
                    std::cout << "  [loss] " << name << " (numel=" << cn << ") v0=" << n->data()[0]
                              << " full_sum=" << s << std::endl;
                }
                else
                    std::cout << "  [loss] " << name << " = (null/无值)" << std::endl;
            };
            print_loss("fape", loss_fape_node);
            print_loss("chi", loss_chi_node);
            print_loss("distogram", loss_distogram_node);
            print_loss("msa", loss_msa_node);
            print_loss("conf", loss_conf_node);
            std::cout << "  [loss] total = " << batch_loss << std::endl;
            // 手动重算 total（用 loss 分量 data()），对比 graph total，判断 total 诊断 nan 是否 buffer 复用假象
            {
                auto v1 = [](TensorF32* n){ return (n && n->data() && n->numel()==1) ? n->data()[0] : 0.0f; };
                double manual = 0.5*v1(loss_fape_node) + 0.5*v1(loss_chi_node) + 0.3*v1(loss_distogram_node)
                              + 2.0*v1(loss_msa_node) + 0.01*v1(loss_conf_node);
                std::cout << "  [loss] manual_total = " << manual << std::endl;
            }
            // ---- chi 诊断：值侧重算 torsion/norm，检查 chi_mask 与 gt 是否正常 ----
            if (getenv("GRAPH_DEBUG_LOSS") && go.alpha && go.alpha->numel() > 0) {
                const int64_t chN = input.gt_chi.shape().dims[0]*input.gt_chi.shape().dims[1];
                const float* ap = go.alpha->data();          // (B,L,7,2) row-major, B=1
                const float* gtp = input.gt_chi.data();       // (B,L,7,2)
                const float* cmp = input.chi_mask.data();     // (B,L,7)
                double sum_sqdiff=0, sum_mask=0; bool mask_neg=false; long gt_zero_cnt=0;
                double npos_chi=0, npos_norm=0;
                for (int64_t r=0; r<chN; ++r) {
                    for (int a=0; a<7; ++a) {
                        float m = cmp[r*7+a];
                        if (m < 0) mask_neg=true;
                        if (m > 0) {
                            double p0=ap[r*14+a*2], p1=ap[r*14+a*2+1];
                            double g0=gtp[r*14+a*2], g1=gtp[r*14+a*2+1];
                            double rn=sqrt(p0*p0+p1*p1+1e-8);
                            double n0=p0/rn, n1=p1/rn;
                            double sd=(n0-g0)*(n0-g0)+(n1-g1)*(n1-g1);
                            sum_sqdiff += sd*m; sum_mask += m;
                            if (g0==0&&g1==0) ++gt_zero_cnt;
                        }
                    }
                }
                std::cout << "  [chi_diag] mask_neg="<<mask_neg<<" gt_zero="<<gt_zero_cnt
                          << " sum_mask="<<sum_mask<<" value_side_torsion="
                          << (sum_mask>0?sum_sqdiff/sum_mask:0.0) << " ap[0..3]="
                          << ap[0]<<","<<ap[1]<<","<<ap[2]<<","<<ap[3]<<std::endl;
            }
            // buffer 地址诊断：若不同 loss 的 data() 指针相同 → Gallocr buffer 复用覆盖
            std::cout << "  [bufaddr] fape=" << (void*)(loss_fape_node?loss_fape_node->data():nullptr)
                      << " msa=" << (void*)(loss_msa_node?loss_msa_node->data():nullptr)
                      << " chi=" << (void*)(loss_chi_node?loss_chi_node->data():nullptr)
                      << " dist=" << (void*)(loss_distogram_node?loss_distogram_node->data():nullptr)
                      << " conf=" << (void*)(loss_conf_node?loss_conf_node->data():nullptr) << std::endl;
            // total 的 src 链（scale+add）回溯，定位 nan
            if (total_node) {
                TensorF32* cur = total_node;
                for (int depth = 0; cur && depth < 8; depth++) {
                    double s = 0; int64_t cn = cur->numel();
                    if (cur->data()) for (int64_t q = 0; q < cn; q++) s += cur->data()[q];
                    std::cout << "  [total-chain] d=" << depth << " op=" << cur->op
                              << " numel=" << cn << " sum=" << s << " addr=" << (void*)cur->data()
                              << (cn > 0 && cur->data() ? " v0=" + std::to_string(cur->data()[0]) : "")
                              << " src1_sum=" << (cur->src[1] && cur->src[1]->data() ?
                                  std::to_string([&](){double ss=0; for(int64_t q=0;q<cur->src[1]->numel();q++) ss+=cur->src[1]->data()[q]; return ss;}()) : "null")
                              << " src1_addr=" << (void*)(cur->src[1] ? cur->src[1]->data() : nullptr)
                              << std::endl;
                    cur = cur->src[0];
                }
            }
            std::cout << "  [loss] msa_logits=" << (go.msa_logits ? "node" : "null")
                      << " distogram=" << (go.distogram ? "node" : "null") << std::endl;
            if (go.msa_logits && go.msa_logits->data()) {
                int64_t ne = go.msa_logits->numel();
                std::cout << "  [msa_logits] numel=" << ne << " dims=["
                          << go.msa_logits->shape().dims[0] << "," << go.msa_logits->shape().dims[1] << ","
                          << go.msa_logits->shape().dims[2] << "," << go.msa_logits->shape().dims[3] << "]";
                if (ne > 0) std::cout << " v[0..4]=" << go.msa_logits->data()[0] << ","
                                      << go.msa_logits->data()[1] << "," << go.msa_logits->data()[2] << ","
                                      << go.msa_logits->data()[3] << "," << go.msa_logits->data()[4];
                std::cout << std::endl;
                // 从 msa_logits 回溯 src 链，定位输出 0 的节点
                TensorF32* cur = go.msa_logits;
                for (int depth = 0; cur && depth < 8; depth++) {
                    double s = 0; int64_t cn = cur->numel();
                    if (cur->data()) for (int64_t q = 0; q < cn; q++) s += cur->data()[q];
                    std::cout << "  [chain] d=" << depth << " op=" << cur->op
                              << " numel=" << cn << " sum=" << s
                              << (cn > 0 && cur->data() ? " v0=" + std::to_string(cur->data()[0]) : "")
                              << std::endl;
                    cur = cur->src[0];
                }
            }
            if (input.bert_mask.numel() > 0) {
                std::cout << "  [bert_mask] numel=" << input.bert_mask.numel()
                          << " sum=" << [&](){ double s=0; const float* d=input.bert_mask.data();
                              for (int64_t i=0;i<input.bert_mask.numel();i++) s+=d[i]; return s; }()
                          << std::endl;
            }
            // 参数统计：PARAM 节点数、有 grad 的 PARAM 数（定位 params=0）
            int n_param=0, n_param_grad=0, n_all_grad=0;
            for (int i=0;i<cgraph->n_nodes();i++) {
                TensorF32* nn = cgraph->graph_node(i);
                if (nn->flag & TENSOR_FLAG_PARAM) {
                    n_param++;
                    if (cgraph->graph_get_grad(nn)) n_param_grad++;
                }
                if (cgraph->graph_get_grad(nn)) n_all_grad++;
            }
            std::cout << "  [param] total=" << n_param << " with_grad=" << n_param_grad
                      << " any_node_with_grad=" << n_all_grad << std::endl;
            // 参数值统计：抽查前几个 PARAM 节点 data() 是否非 0（判断 weight 是否初始化）
            {
                int shown = 0;
                for (int i = 0; i < cgraph->n_nodes() && shown < 3; i++) {
                    TensorF32* nn = cgraph->graph_node(i);
                    if (nn->flag & TENSOR_FLAG_PARAM && nn->data()) {
                        double s = 0; int64_t nn2 = nn->numel();
                        for (int64_t q = 0; q < nn2; q++) s += nn->data()[q];
                        std::cout << "  [param-val] numel=" << nn2 << " sum=" << s
                                  << " v[0..2]=" << (nn2>0?nn->data()[0]:0) << ","
                                  << (nn2>1?nn->data()[1]:0) << ","
                                  << (nn2>2?nn->data()[2]:0) << std::endl;
                        shown++;
                    }
                }
            }
            // loss 节点结构诊断
            auto print_loss_node = [&](const char* name, TensorF32* n) {
                if (!n) { std::cout << "  [lossnode] " << name << " = null" << std::endl; return; }
                std::cout << "  [lossnode] " << name << " op=" << n->op << " numel=" << n->numel()
                          << " ndim=" << n->shape().ndim()
                          << " dims=[" << n->shape().dims[0] << "," << n->shape().dims[1] << ","
                          << n->shape().dims[2] << "," << n->shape().dims[3] << "]"
                          << " src0_op=" << (n->src[0] ? n->src[0]->op : -1)
                          << " src1_op=" << (n->src[1] ? n->src[1]->op : -1)
                          << std::endl;
            };
            print_loss_node("fape", loss_fape_node);
            print_loss_node("msa", loss_msa_node);
            print_loss_node("distogram", loss_distogram_node);
            // msa loss (div) 的 src 链回溯
            if (loss_msa_node) {
                TensorF32* cur = loss_msa_node;
                for (int depth = 0; cur && depth < 6; depth++) {
                    double s = 0; int64_t cn = cur->numel();
                    if (cur->data()) for (int64_t q = 0; q < cn; q++) s += cur->data()[q];
                    std::cout << "  [msa-loss-chain] d=" << depth << " op=" << cur->op
                              << " numel=" << cn << " sum=" << s
                              << (cn > 0 && cur->data() ? " v0=" + std::to_string(cur->data()[0]) : "")
                              << std::endl;
                    cur = cur->src[0];
                }
            }
            // 输入诊断：msa_latent / true_msa / true_coords 是否有效
            auto print_vals = [](const char* name, const TensorF32& t) {
                const float* d = t.data();
                double s = 0; int64_t nn = t.numel();
                float mn = 1e30f, mx = -1e30f; int64_t nnan = 0;
                for (int64_t i = 0; i < nn; i++) { float v = d[i]; if (v != v) { nnan++; continue; } s += v; if (v<mn) mn=v; if (v>mx) mx=v; }
                std::cout << "  [input] " << name << " numel=" << nn << " sum=" << s
                          << " min=" << (nn?mn:0) << " max=" << (nn?mx:0) << " nnan=" << nnan;
                if (nn > 0) std::cout << " v[0..3]=" << d[0] << "," << d[1] << "," << d[2] << "," << d[3];
                std::cout << std::endl;
            };
            print_vals("msa_latent", input.msa_latent);
            print_vals("true_msa", input.true_msa);
            print_vals("bert_mask", input.bert_mask);
            print_vals("coords", input.coords);
            print_vals("true_coords", input.true_coords);
            print_vals("D_onehot", input.D_onehot);
            print_vals("pair_mask", input.pair_mask);
            print_vals("ca_mask", input.ca_mask);
        }

        epoch_loss += batch_loss;
        
        // ===== 计时代码: 汇总 epoch 耗时 =====
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            epoch_end - epoch_start).count();
        double epoch_sec = static_cast<double>(epoch_ms) / 1000.0;

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                  << " completed in " << epoch_ms << " ms"
                  << " (forward " << fwd_ms << " ms)"
                  << ", loss: " << batch_loss
                  << ", grad_norm: " << grad_norm << std::endl;

        // ============================================================
        // Checkpoint 保存: 每隔 ckpt_interval 个 epoch, 以及最后一个 epoch
        // (ckpt_interval=0 时禁用保存, 避免除零)
        // ============================================================
        const bool is_ckpt_epoch =
            (ckpt_interval > 0) &&
            (((epoch + 1) % ckpt_interval == 0) || (epoch + 1) == num_epochs);
        if (is_ckpt_epoch) {
            // 确保输出目录存在
            std::error_code ec;
            std::filesystem::create_directories(ckpt_dir, ec);

            std::string path = ckpt_dir + "/ckpt_epoch" + std::to_string(epoch + 1) + ".gguf";
            std::cout << "  [Checkpoint] Saving at epoch " << epoch + 1
                      << "/" << num_epochs
                      << " (loss=" << batch_loss
                      << ", epoch_time=" << epoch_ms << " ms) ... ";

            auto ckpt_start = std::chrono::high_resolution_clock::now();
            try {
                // before save: use backend->synchronize force cpu to wait all gpu work finish
                // cudaStreamSynchronize or cudaDeviceSynchronize
                save_checkpoint(model, path, epoch, num_epochs, batch_loss, epoch_sec);
                auto ckpt_end = std::chrono::high_resolution_clock::now();
                auto ckpt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   ckpt_end - ckpt_start).count();

                // 打印保存后的文件大小
                std::uintmax_t fsize = 0;
                if (!ec) fsize = std::filesystem::file_size(path);
                std::cout << "OK -> " << path
                          << " (" << fsize << " B, save took " << ckpt_ms << " ms)"
                          << std::endl;

                // 开发功能: 统计所有权重的内存占用 (在 save_checkpoint 内用同款 CPU 拷贝逻辑)
                estimate_params_memory(model);
            } catch (const std::exception& e) {
                std::cerr << "  [Checkpoint] FAILED: " << e.what() << std::endl;
            }
        }
    }
    
    // 6. 保存最终模型 (仍保留原有 .bin 保存, 以便兼容)
    model.save_weights("ppml_weights.bin");
    
    // 7. 导出 ONNX
    ONNXExportConfig onnx_config;
    onnx_config.output_path = "ppml_model.onnx";
    onnx_config.opset_version = 17;
    
    // 设置动态维度
    onnx_config.dynamic_dims = {
        {"batch", 1, 4},
        {"seq_len", 32, 2048},
        {"n_seq", 1, 512},
        {"n_templ", 1, 4}
    };
    
    /* ONNXExporter exporter;
    exporter.export_model(model, onnx_config);
    
    if (exporter.validate(onnx_config.output_path)) {
        std::cout << "ONNX model exported and validated successfully" << std::endl;
        std::cout << exporter.get_model_info(onnx_config.output_path) << std::endl;
    }
     */
    // 8. 清理
    py.finalize();
    
    return 0;
}
