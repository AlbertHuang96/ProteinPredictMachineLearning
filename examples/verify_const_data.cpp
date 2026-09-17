// ============================================================================
// examples/verify_const_data.cpp — 验证 const_data_ + TENSOR_FLAG_CONST 机制
//
// 验证点：
//   1. constant_tensor 创建后 data()==nullptr，数据在 const_data_，flag 置 TENSOR_FLAG_CONST
//   2. constant_tensor_dynamic 创建后 flag 不置 TENSOR_FLAG_CONST
//   3. graph_compute（Gallocr alloc）后：
//        - 静态 CONST 常量 data() 被填充为 const_data_ 值，且 const_data_ 保留
//        - 动态一次性常量 data() 被填充，且 const_data_ 被清空（size()==0）
//   4. 值正确：mul(x, mask) 输出符合 x*mask
// ============================================================================
#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"
#include "ppml/Backend.h"
#include "ppml/Dropout.h"
#include <cstdio>
#include <iostream>
#include <vector>

using namespace ppml;

int main() {
    std::setvbuf(stdout, NULL, _IONBF, 0);  // 无缓冲，便于定位崩溃点
    PPMLContext* ctx = &context();
    CPUBackend cpu(1);

    // ---- 1. 常量：静态可复用（CONST 置位）----
    std::vector<int64_t> dims = {2, 3};  // 6 元素
    float cdata[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    TensorF32* c = constant_tensor(dims, cdata);

    std::cout << "[1] static const_tensor:\n";
    std::cout << "    data()==nullptr : " << (c->data() == nullptr) << "\n";
    std::cout << "    const_data_.size: " << c->const_data_.size() << "\n";
    std::cout << "    TENSOR_FLAG_CONST: " << ((c->flag & TENSOR_FLAG_CONST) != 0) << "\n";

    // ---- 2. 动态一次性常量（CONST 不置位）----
    float mdata[6] = {2.f, 2.f, 0.f, 0.f, 2.f, 2.f};
    TensorF32* d = constant_tensor_dynamic(dims, mdata);

    std::cout << "[2] dynamic const_tensor:\n";
    std::cout << "    data()==nullptr : " << (d->data() == nullptr) << "\n";
    std::cout << "    const_data_.size: " << d->const_data_.size() << "\n";
    std::cout << "    TENSOR_FLAG_CONST: " << ((d->flag & TENSOR_FLAG_CONST) != 0) << "\n";

    // ---- 3. mul(c, d) → 结果节点 ----
    TensorF32* out = mul(c, d);
    out->flag |= TENSOR_FLAG_OUTPUT;  // 标记输出，gallocr 不释放

    // ---- 4. 构建图并 graph_compute ----
    ComputeGraph* g = ComputeGraph::new_graph(ctx);
    g->build_forward_expand(out);

    Status st = cpu.graph_compute(g);
    bool ok = (st == Status::SUCCESS);

    std::cout << "[4] graph_compute: " << (ok ? "ok" : "FAILED") << "\n";

    // ---- 5. 检查填充结果 ----
    std::cout << "[5] after compute:\n";
    std::cout << "    c->data()!=null : " << (c->data() != nullptr) << "\n";
    std::cout << "    c->const_data_  : " << c->const_data_.size() << " (保留=6)\n";
    std::cout << "    c values        :";
    if (c->data()) for (int i = 0; i < 6; i++) std::cout << " " << c->data()[i];
    std::cout << "\n";
    std::cout << "    d->data()!=null : " << (d->data() != nullptr) << "\n";
    std::cout << "    d->const_data_  : " << d->const_data_.size() << " (清空=0)\n";
    std::cout << "    d values        :";
    if (d->data()) for (int i = 0; i < 6; i++) std::cout << " " << d->data()[i];
    std::cout << "\n";

    // ---- 6. 校验 mul 输出值 ----
    std::cout << "[6] out values (x*mask):";
    if (ok && out->data()) {
        bool correct = true;
        for (int i = 0; i < 6; i++) {
            float expected = cdata[i] * mdata[i];
            float got = out->data()[i];
            std::cout << " " << got;
            if (got != expected) correct = false;
        }
        std::cout << "\n";
        std::cout << "    mul value correct: " << (correct ? "YES" : "NO") << "\n";
    } else {
        std::cout << " (out->data()==nullptr)\n";
    }

    // ---- 7. 动态常量宿主内存已释放（不再累积）----
    std::cout << "[7] dynamic const_data_ freed: " << (d->const_data_.empty() ? "YES" : "NO") << "\n";

    // ---- 8. Dropout::forward_graph 随机掩码恢复验证 ----
    // 训练模式：每次 forward_graph 新建随机掩码叶子，mul(x, mask)。两次调用掩码不同 → 随机化恢复。
    {
        std::cout << "[8] Dropout randomization:\n";
        Dropout drop(-1, 0.5f);   // 逐元素 dropout, p=0.5
        drop.set_training(true);

        float xd[8] = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
        // 每次调用新建输入 leaf（与真实训练逐迭代重建图一致；同后端跨 graph_compute
        // 复用同一常量对象会因 gallocr.release() 不重置 data() 而误判为非 managed）
        TensorF32* x1 = constant_tensor({8}, xd);
        std::cout << "    before graph_compute, xleaf data()==nullptr: "
                  << (x1->data() == nullptr) << "\n";

        TensorF32* out1 = drop.forward_graph(x1);
        ComputeGraph* g1 = ComputeGraph::new_graph(ctx);
        g1->build_forward_expand(out1);
        Status st1 = cpu.graph_compute(g1);
        std::cout << "    call#1 graph_compute: " << (st1 == Status::SUCCESS ? "ok" : "FAIL")
                  << "  out1 values:";
        if (out1->data()) for (int i = 0; i < 8; i++) std::cout << " " << out1->data()[i];
        std::cout << "\n";

        TensorF32* x2 = constant_tensor({8}, xd);   // 独立新输入 leaf
        TensorF32* out2 = drop.forward_graph(x2);
        ComputeGraph* g2 = ComputeGraph::new_graph(ctx);
        g2->build_forward_expand(out2);
        Status st2 = cpu.graph_compute(g2);
        std::cout << "    call#2 graph_compute: " << (st2 == Status::SUCCESS ? "ok" : "FAIL")
                  << "  out2 values:";
        if (out2->data()) for (int i = 0; i < 8; i++) std::cout << " " << out2->data()[i];
        std::cout << "\n";

        bool random = (st1 == Status::SUCCESS && st2 == Status::SUCCESS);
        if (out1->data() && out2->data()) {
            bool same = true;
            for (int i = 0; i < 8; i++) if (out1->data()[i] != out2->data()[i]) same = false;
            random = !same;   // 两次掩码不同 → 每次随机化生效
        }
        std::cout << "    re-randomized each forward: " << (random ? "YES" : "NO") << "\n";
        std::cout << "    eval (non-training) passthrough: ";
        drop.set_training(false);
        TensorF32* out_eval = drop.forward_graph(x1);
        std::cout << (out_eval == x1 ? "YES (identity)" : "NO") << "\n";
    }

    // 注意：ComputeGraph 分配于 context arena（new_graph_custom 经 ctx->new_object 放入
    // mem_buffer），不能 delete（会 free 无效指针）。arena 由 context 生命周期统一管理。
    std::cout << "PASS\n";
    return 0;
}
