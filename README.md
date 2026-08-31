
# ProteinPredictMachineLearning Project
PPML project  
Inspired by RosettaFoldAllAtom(RFAA)
and   
llama.cpp and GGML  

## 训练基准 (Training Benchmark)

### Small setting CPU training  
Env: Intel i5-1335U 12 Core / 15 GB RAM / Pure CPU (LD_PRELOAD libstdc++)

小配置纯 CPU 训练 5 epoch (P62891, L=51, MSA_DEPTH=8, N_EXTRA=1, N_MAIN=2, N_REFINE=1, DEV_SE3=1, SE3_TOPO=per_block 开关B)  
环境: Intel i5-1335U 12 核 / 15 GB RAM / 纯 CPU (LD_PRELOAD 系统 libstdc++)

| Epoch | 耗时 (ms) | forward (ms) | loss | grad_norm |
|-------|----------|--------------|------|-----------|
| 1/5   | 34642    | 25576        | 14.16    | 0.11 |
| 2/5   | 35859    | 25254        | 15.1777  | 0.11 |
| 3/5   | 43021    | 28667        | 14.7508  | 0.11 |
| 4/5   | 38636    | 27085        | 14.3692  | 0.11 |
| 5/5   | 37338    | 25205        | 14.3877  | 0.11 |

- 5 epoch finished with EXIT=0，loss limited (no NaN)，grad_norm stable=0.11
- 5 epoch 全部完成 EXIT=0，loss 全有限（无 NaN），grad_norm 稳定 0.11


### Small setting hybrid training 5 epoch
Env: Intel i5-1335U 12 Core / 15 GB RAM / NVIDIA GeForce RTX 2050 4GB (compute 8.6)

小配置混合 CUDA 训练 5 epoch (P62891, L=51, MSA_DEPTH=8, N_EXTRA=1, N_MAIN=2, N_REFINE=1, DEV_SE3=1, SE3_TOPO=per_block 开关B)  
环境: Intel i5-1335U 12 核 / 15 GB RAM / NVIDIA GeForce RTX 2050 4GB (compute 8.6, Tensor Core YES) / 混合调度 (BackendScheduler, GPU scatter 开启, 未设 PPML_CUDA_NO_SCATTER)

| Epoch | 耗时 (ms) | forward (ms) | loss | grad_norm |
|-------|----------|--------------|------|-----------|
| 1/5   | 25972    | 24582        | 0.00      | 0.11 |
| 2/5   | 25712    | 23931        | 2.85e-05  | 0.11 |
| 3/5   | 25346    | 23699        | 2.85e-05  | 0.11 |
| 4/5   | 25045    | 23489        | 2.85e-05  | 0.11 |
| 5/5   | 25942    | 24138        | 2.85e-05  | 0.11 |

GPU stat（nvidia-smi 1s sample，112 sample points）: VRAM peak 3923/4096 MiB (95.8%), usage peak 99%, avg usage 2.8%, nonzero sample 8.9%
显存峰值 3923/4096 MiB (95.8%) / 利用率峰值 99% / 平均利用率 2.8% / 非零采样占比 8.9%

- 5 epoch finished EXIT=0，loss limited, grad_norm stable=0.11
- 5 epoch 全部完成 EXIT=0，loss 全有限（无 NaN/Inf，0.00 与 2.85e-05 为小配置 loss 分量特征），grad_norm 稳定 0.11

- Mixed mode: about 25.7s per epoch; About 1.47x faster than CPU mode
- 混合 CPU+GPU：每 epoch ~25.7s（CPU 纯跑 ~37.8s，提速约 1.47×）；GPU 平均利用率低（2.8%）因主要负载仍落 CPU（CPU buffer 峰值 4.35GB vs CUDA 1.11GB）


dev/training data:

data/training_batch_data/P62891_alignment.a3m data/P62891.fasta  

## Project Structure

Main goal:  

predict the 3D structure of a protein with limited compute resources,   
i.e. personal computer  
still leave potential to running on a GPU server  
or multiplatform deploy  
Theoretically, it could predict structure of a protein including side-chain and all-atom in the near future :)

### Data pipeline
preprocessing
inputs:
MSA
templates

### Transformers

forward and backend process  
backend graph  
loss functions  
GGUF saving and loading  

### Optimize (TODOs)

LoRA fine-tuning  

GPU backend i.e. CUDA node  

detect hardware:  
kv-cache  
paged attention?  

quantized data training?  
mixed precision training?  



## Project Architecture

```
RFAA-Cpp
│
├── Core Infrastructure ───── Tensor + Memory Pool + Compute Graph Engine
│   ├── Tensor abstraction         GGML-style tensor
│   ├── Context memory pool       Unified allocation for all weights & intermediate tensors
│   ├── ComputeGraph              DAG compute graph with forward expansion + reverse autograd
│   └── Backend scheduler         Multi-backend (CPU / CUDA) automatic graph splitting & dispatch
│
├── Embedding ───── Input features → model representations
│   ├── EmbeddingLayer            MSA token embedding
│   ├── BondEmbedding             Chemical bond type embedding (8 bond types → pair)
│   ├── LinearLayer / LayerNorm   Fully-connected + layer normalization
│   └── PositionalEncoding        Relative positional encoding (distance + bond features)
│
├── Attention Modules ───── Sequence and pairwise information exchange
│   ├── MSA Attention             Row Attention + Col Attention + Global Col Attention
│   ├── Pair Attention            Row / Col Attention (with State bias)
│   ├── Cross Attention           Template information injection (State ← Template)
│   ├── TriangleMultiplication    Outgoing / Incoming (pair information propagation)
│   ├── FeedForward               Feed-forward networks (MSA / Pair)
│   └── TemplatePairStack         Template pair stack processing
│
├── Track System ───── 1D / 2D / 3D multi-pathway information flow
│   ├── MSATrack  (1D)           MSA representations (D=256), sequence dimension
│   ├── PairTrack  (2D)          Residue-pair representations (D=128), spatial dimension
│   └── StateTrack (1D)          Structure state (D=32), 3D coords → structural features
│
├── SE(3) Equivariant Network ───── Group-theory constrained 3D GNN
│   ├── Math utilities            Spherical harmonics / Clebsch-Gordan coeffs / Wigner D matrices
│   ├── SE3Basis                  Precomputed spherical harmonic basis + CG coupling (cached)
│   ├── RadialFunc                Learnable radial profile function (MLP)
│   ├── SE3 Convolution           Partial conv → 1×1 channel mixing → multi-head self-attention
│   └── SE3Transformer            Multi-layer equivariant residual blocks (rotation/translation invariant)
│
├── Model Layer ───── End-to-end prediction pipeline
│   ├── IterBlock                Core iteration block (Attention → TriangleMul → SE3 → Structure update)
│   ├── RefineBlock              Refinement block (complex node/edge embeddings)
│   ├── FullBlock                Full MSA block (with global column attention)
│   └── RFAAModel                 Main model: Embedding → Blocks ×16 → Output Heads
│
├── Data Pipeline ───── Input preparation
│   ├── A3M parsing        MSA and template search result files
│   ├── Template feature extraction   1D features + 2D features + coordinates
│   ├── 
│   └── RFAADataLoader           End-to-end preprocessing (sequence → model input)
│
├── Serialization & Integration
│   ├── GGUF model format         Weight I/O + metadata + quantization type support
│   └── Python bridge             pybind11 bidirectional interop (Python ↔ C++)
│
├── GPU Acceleration (CUDA)
│   ├── CUDA Tensor               Device-side tensor implementation
│   ├── 
│
└── Build Artifacts
    ├── rfaa_core                 Core library (static / shared)
    ├── rfaa_gpu                  CUDA acceleration backend (DLL)
    ├── rfaa_python               Python bridge library (DLL / pybind)
    └── rfaa_train / rfaa_infer   Training + inference executables
```

### Key Design Features

| Feature | Description |
|------|------|
| GGML-style memory management | Context-based memory pool; all weights and intermediate tensors have controlled lifetimes, no fragmentation |
| Compute graph + eager execution | `forward_graph` builds graph (training/autograd), `forward_exec` runs directly (inference) |
| Multi-backend scheduling | BackendScheduler auto-assigns graph nodes to CPU / CUDA with cross-device data transfer |
| SE(3) equivariance | Group-theoretic spherical harmonics + CG coefficients + Wigner D matrices ensure rotation/translation physical consistency |

| Track architecture | MSA (1D) / Pair (2D) / Coords(3D) three-pathway information flow, inspired by AlphaFold2 / RosettaFold |
| Template-aware | Full injection of template 1D/2D/3D features via CrossAttention + TemplatePairStack |
| Modular iteration | IterBlock / RefineBlock / FullBlock encapsulate update logic at different granularity for flexible composition |

## Build Requirements

- CMake >= 3.18
- CUDA >= 11.7
- Python >= 3.8 (with development headers)
- ONNX Runtime >= 1.15

## Build Commands

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DBUILD_PYTHON_BRIDGE=ON
make -j$(nproc)
```

## TODO

- Replace all graph node constructors
- Move modules into models
- Homo-oligomer support
- residx 目前是理想化连续索引 0..L-1，未用 CSV/PDB 的真实 ResNum；单链连续场景够用，但多链/缺残基(gap)/非标准编号时会丢失真实序列间隔信息（gap 应拉开但当前视为相邻）。CSV 已有 PDB_ResNum 列，尚未用于构建 residx。
- **模板 pair 注入 T>1（待实现）**：`PPMLModel::forward_graph` 的模板注入中，state 分支已用全部 T 模板作 cross-attn key；但 pair 分支当前**仅用 t=0 单模板**（值版 `PairTrack::inject_template` 亦是 T=1 语义）。原因：图基础设施 4D 上限（`kernel_concat`/`kernel_permute` 只支持 ≤4D），且缺沿 dims[3] 的归约 op（`sum`/`mean` 只做全归约，`sum_rows` 只沿 dims[1]）。完整方案：① 新增沿任意维的 reduce op；或 ② 把 templ_pair `[64,L,L,T]` 经 permute 重排为 `[64,1,1,L*L*T]` 作为多模板 kv（T 折叠进 key 长度），query 仍为 B*L*L。需同步值版语义。

### Torsion Indices Reference
- Negative index indicates the previous residue
- Order:
  - omega/phi/psi: 0-2
  - chi_1-4 (prot): 3-6
  - cb/cg bend: 7-9
  - eps(p)/zeta(p): 10-11
  - alpha/beta/gamma/delta: 12-15
  - nu2/nu1/nu0: 16-18
  - chi_1 (na): 19


```
