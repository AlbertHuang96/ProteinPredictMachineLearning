#include "rfaa/Model.h"
#include "rfaa/DataLoader.h"
#include "rfaa/ComputeGraph.h"
#include "rfaa/Context.h"
#include "rfaa/PythonBridge.h"
#include "rfaa/ONNXExporter.h"
#include "rfaa/GradientClipper.h"
#include "rfaa/GGUF.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>

using namespace rfaa;

// ============================================================================
// 工具: 将值张量 (ModelOutput.coords / ModelInput.true_coords) 包装为图节点 leaf
// 注意: 当前 RFAAModel::forward 基于数值张量计算, 尚未接入 autograd 图。
//       这里将 pred/true coords 拷入 context leaf 节点, 使 fape_loss 等图节点
//       损失函数可以正确构造计算图。待模型接入图后端后, 此包装可移除。
// ============================================================================
TensorF32* wrap_value_as_leaf(const TensorF32& t, const std::vector<int64_t>& dims) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size() && i < 4; i++) ne[i] = dims[i];
    TensorF32* leaf = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
    // 数据按扁平顺序拷入. Tensor 是 move-only (禁拷贝), 故:
    //   - CUDA 上: 用 t.cpu() 生成新的 CPU 张量 (可移动) 再取 data
    //   - CPU 上:  直接取 t.data() 拷入
    if (t.device() == Device::CUDA) {
        TensorF32 tcpu = t.cpu();  // 移动构造, 合法
        std::memcpy(leaf->data(), tcpu.data(), tcpu.numel() * sizeof(float));
    } else {
        std::memcpy(leaf->data(), t.data(), t.numel() * sizeof(float));
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
size_t estimate_params_memory(const RFAAModel& model) {
    // params() 收集的指针顺序固定, 但此处通过尺寸计算内存;
    // 由于 params() 为非 const, 这里用一个 const 转换不了, 故通过 const_cast 调用
    // (仅读取 nbytes/shape, 不修改任何状态, 安全)
    auto& m = const_cast<RFAAModel&>(model);
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
void save_checkpoint(RFAAModel& model, const std::string& path,
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
                  {"arch", "rfaa_v1"},
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
    
    // 2. 创建模型
    RFAAConfig config;
    config.d_msa = 256;
    config.d_pair = 128;
    config.d_state = 32;
    config.n_extra_blocks = 4;
    config.n_main_blocks = 8;
    config.n_refine_blocks = 4;
    
    RFAAModel model(config);
    
    // 3. 转移到 GPU
    model.to(Device::CUDA);
    model.train();
    
    std::cout << "Model created and moved to CUDA" << std::endl;
    
    // 4. 加载预训练权重 (通过 Python 桥接)
    if (argc > 1) {
        std::string weights_path = argv[1];
        ProteinTools::load_torch_weights(model, weights_path);
    }
    
    // ============================================================
    // 数据加载: 通过 RFAADataLoader 从 A3M/HHR + CSV mapping 加载
    // CSV 提供真实坐标 (true_coords), 作为 FAPE 的 ground truth
    // ============================================================
    // 命令行参数: train <a3m> <hhr> <sequence> <csv_mapping>
    // 例如: train 1a00.a3m 1a00.hhr "MVLSPADKTNVKAAWGKVG..." data/1a00_P69905_mapping.csv
    std::string a3m_path  = (argc > 2) ? argv[2] : "query.a3m";
    std::string hhr_path  = (argc > 3) ? argv[3] : "query.hhr";
    std::string sequence  = (argc > 4) ? argv[4] : "";
    std::string csv_path  = (argc > 5) ? argv[5] : "data/1a00_P69905_mapping.csv";

    if (sequence.empty()) {
        // 无显式序列时从 CSV 行数推断长度 (demo 场景)
        std::cerr << "No sequence provided; using CSV-derived length. "
                     "Pass <a3m> <hhr> <sequence> <csv> as argv[2..5]." << std::endl;
    }

    RFAADataLoader loader("", "", 512, 4, 2048);
    ModelInput input = loader.load_from_files(a3m_path, hhr_path, sequence, csv_path);
    int L = 0;
    if (!sequence.empty()) {
        L = static_cast<int>(sequence.length());
    } else if (input.true_coords.numel() > 0) {
        L = static_cast<int>(input.true_coords.shape().dims[1]);
    }

    std::cout << "Loaded data. L=" << L
              << " true_coords shape=(" << (input.true_coords.numel() > 0 ? 1 : 0)
              << "," << L << ",3,3)" << std::endl;

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
    const int num_epochs = 10;

    // ---- Checkpoint 配置 ----
    // 每隔 checkpoint_interval 个 epoch 保存一次权重 (GGUF)。
    // TODO(增强): 增加命令行参数 --ckpt-interval N 覆盖此默认值,
    //             例: 解析 argc/argv 后 ckpt_interval = atoi(argv[k])。
    int ckpt_interval = 2;                     // 默认每 2 个 epoch 保存一次
    const std::string ckpt_dir = "checkpoints"; // checkpoint 输出目录

    // 可选: 开发期统计所有权重的内存占用 (不写文件)
    // 若要打印初始权重占用的内存, 取消下面一行注释:
    // size_t init_param_bytes = estimate_params_memory(model);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ===== 计时代码: epoch 级 + 前向/损失阶段子计时 =====
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        float epoch_loss = 0.0f;
        
        // 前向传播
        auto fwd_start = std::chrono::high_resolution_clock::now();
        auto output = model.forward(input);
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
        TensorF32 pred_flat = output.coords.view({N_atoms, 3});
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

        // 2) Chi (扭转角) loss — 当前无真实扭转角监督, 用标量 0 占位
        //    接入真实 gt 时替换为: supervised_chi_loss(unnormed, gt, chi_mask, seq_mask)
        TensorF32* loss_chi_node       = loss(constant_scalar(0.0f));

        // 3) Distogram loss — 当前无 distogram 标签, 用标量 0 占位
        TensorF32* loss_distogram_node = loss(constant_scalar(0.0f));

        // 4) Masked MSA loss — 当前无 MSA 掩码监督, 用标量 0 占位
        TensorF32* loss_msa_node       = loss(constant_scalar(0.0f));

        // 5) Confidence (pLDDT) loss — 当前无 LDDT 标签, 用标量 0 占位
        TensorF32* loss_conf_node      = loss(constant_scalar(0.0f));

        // --- total_loss: 0.5*FAPE + 0.5*Chi + 0.3*Distogram + 2.0*MSA + 0.01*Conf ---
        TensorF32* total_node = loss(total_loss(
            loss_fape_node, loss_chi_node, loss_distogram_node, loss_msa_node, loss_conf_node));

        // ============================================================
        // 构建 backward 计算图并执行 (全局裁剪)
        // ============================================================
        RFAAContext* ctx = &context();
        ComputeGraph* cgraph = ComputeGraph::new_graph(ctx);
        cgraph->build_forward_expand(total_node);
        cgraph->build_backward_expand(ctx, nullptr);

        // 若已接入 backend, 可在此调用 graph_compute + clip_grad_norm + optimizer
        // backend->graph_compute(cgraph);
        // float grad_norm = clip_grad_norm(cgraph, 0.1f);

        // 读取 total loss 数值 (FAPE 项已正确构造, 其余项为 0)
        // 注: 需要 backend->graph_compute 后 total_node->data() 才有值
        float batch_loss = 0.0f;
        if (total_node->data() != nullptr && total_node->numel() == 1) {
            batch_loss = total_node->data()[0];
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
                  << ", loss: " << batch_loss << std::endl;

        // ============================================================
        // Checkpoint 保存: 每隔 ckpt_interval 个 epoch, 以及最后一个 epoch
        // ============================================================
        bool is_ckpt_epoch = (epoch + 1) % ckpt_interval == 0 || (epoch + 1) == num_epochs;
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
    model.save_weights("rfaa_weights.bin");
    
    // 7. 导出 ONNX
    ONNXExportConfig onnx_config;
    onnx_config.output_path = "rfaa_model.onnx";
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
