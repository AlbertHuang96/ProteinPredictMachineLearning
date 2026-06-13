# RFAA C++ 蛋白质学习框架

## 项目结构

```
RFAA-Cpp/
├── CMakeLists.txt              # 主构建配置
├── cmake/
│   └── FindCUDA.cmake          # CUDA 查找模块
├── include/
│   └── rfaa/
│       ├── Core.h              # 核心类型定义
│       ├── Tensor.h            # 张量抽象层
│       ├── Model.h             # 模型接口
│       ├── Track.h             # Track 基类
│       ├── MSA.h               # MSA Track
│       ├── Pair.h              # Pair Track
│       ├── State.h             # State Track
│       ├── Attention.h         # Attention 模块
│       ├── SE3Transformer.h    # SE3 Transformer
│       ├── ONNXExporter.h      # ONNX 导出
│       └── PythonBridge.h      # Python 桥接
├── src/
│   ├── core/
│   │   ├── Tensor.cpp          # CPU Tensor 实现
│   │   └── MemoryPool.cpp      # 内存池管理
│   ├── cuda/
│   │   ├── CudaTensor.cu       # CUDA Tensor 实现
│   │   ├── AttentionKernel.cu  # Attention CUDA 核
│   │   └── SE3Kernel.cu        # SE3 CUDA 核
│   ├── model/
│   │   ├── Embedding.cpp       # Embedding 层
│   │   ├── IterBlock.cpp       # 迭代块
│   │   └── RFAA.cpp            # 主模型
│   ├── tracks/
│   │   ├── MSA.cpp             # MSA Track 实现
│   │   ├── Pair.cpp            # Pair Track 实现
│   │   └── State.cpp           # State Track 实现
│   ├── modules/
│   │   ├── Attention.cpp       # Attention 实现
│   │   ├── TriangleMul.cpp     # Triangle Multiplication
│   │   └── SE3Transformer.cpp  # SE3 Transformer
│   ├── onnx/
│   │   └── ONNXExporter.cpp    # ONNX 导出实现
│   └── python/
│       └── PythonBridge.cpp    # Python C API 桥接
├── core_lib/                   # 核心功能库 (静态/动态)
│   ├── CMakeLists.txt
│   └── ...
├── gpu_backend/                # GPU 加速后端 (DLL)
│   ├── CMakeLists.txt
│   └── ...
├── python_bridge/              # Python 桥接 (DLL)
│   ├── CMakeLists.txt
│   └── ...
├── examples/
│   ├── train.cpp               # 训练示例
│   ├── infer.cpp               # 推理示例
│   └── export_onnx.cpp         # ONNX 导出示例
├── tests/
│   └── ...
└── third_party/
    ├── pybind11/               # C++ Python 绑定
    └── onnxruntime/            # ONNX Runtime
```

## 构建要求

- CMake >= 3.18
- CUDA >= 11.7
- Python >= 3.8 (含开发头文件)
- ONNX Runtime >= 1.15

## 构建命令

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DBUILD_PYTHON_BRIDGE=ON
make -j$(nproc)
```

todo:
替换所有的图节点构造函数
move modules into models

homo-oligomers

# resolve torsion indices
        #  a negative index indicates the previous residue
        # order:
        #    omega/phi/psi: 0-2
        #    chi_1-4(prot): 3-6
        #    cb/cg bend: 7-9
        #    eps(p)/zeta(p): 10-11
        #    alpha/beta/gamma/delta: 12-15
        #    nu2/nu1/nu0: 16-18
        #    chi_1(na): 19

#road map:

flash attention

Inference:
KV cache

ggml:
struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
    memcpy(tensor_a->data, matrix_A, ggml_nbytes(tensor_a));
    memcpy(tensor_b->data, matrix_B, ggml_nbytes(tensor_b));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor *result = ggml_mul_mat(ctx, tensor_a, tensor_b);

    ggml_build_forward_expand(gf, result);

    int n_threads = 1;
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
