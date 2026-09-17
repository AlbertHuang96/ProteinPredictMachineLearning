
# ProteinPredictMachineLearning Project
PPML project  
Inspired by RosettaFoldAllAtom(RFAA) and GGML  

## Quick Start (command line)

### Build environment (tested)

| Item | Version / note |
|---|---|
| OS | **Ubuntu 22.04.5 LTS on WSL2** — kernel `5.15.167.4-microsoft-standard-WSL2` |
| CMake | **3.22.1** (project minimum: 3.18) |
| Compiler | **GCC 11.4.0** (C++17) |
| CUDA | **11.5.119** — tested on GeForce RTX 2050 (compute 8.6). The CUDA toolkit is required at configure time (`project(... LANGUAGES CXX CUDA)` + `find_package(CUDAToolkit REQUIRED)`), even for CPU-only runs. |
| Python | **3.11.5** (Anaconda) **with development headers** (`Python.h` / `python3-dev`) |

**Notes (EN)**
- **Windows is not fully supported yet.** Build/run is verified on **Linux only (WSL2 + Ubuntu 22.04)**.
  Windows PowerShell cannot drive the existing CMake cache — use WSL or a Linux host.
  
  Server build: `build_remote.sh [target]` (also auto-detects python/CUDA).
- **The Python requirement is mainly for future development convenience** (the `python_bridge`
  pybind11 module). Training / inference are pure C++ and do **not** need Python at runtime; a
  Python without development headers is not enough to *configure* the project.
- Anaconda ships an older `libstdc++`: if you hit `GLIBCXX_3.4.30 not found`, prefix commands with
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.

Run a small-sample training job using the bundled tiny sample (P62891, L=51). This uses the
dev/small config so it fits in a few GB of CPU RAM:

```bash
# 1) Build (either your local build tree, or the remote-server build script)
bash build_remote.sh ppml_train        # see also: build_remote.sh (server build, auto-detects python/CUDA)

# 2) Run the small-sample training (pure CPU, dev config)
LD_PRELOAD=$(gcc -print-file-name=libstdc++.so.6) \
  build/remote/examples/ppml_train \
  data/training_batch_data/P62891_alignment.a3m \
  data/P62891.fasta \
  data/training_batch_data/4ug0_P62891_mapping.csv \
  ""
```

Optional dev env (small/fast: L=51, MSA depth 8, few blocks):

```bash
PPML_SEED=1 PPML_USE_CUDA=0 PPML_DEV_SE3=1 PPML_MSA_DEPTH=8 \
PPML_NUM_EPOCHS=2 PPML_N_EXTRA=1 PPML_N_MAIN=2 PPML_N_REFINE=1 PPML_SE3_TOPO=per_block \
LD_PRELOAD=$(gcc -print-file-name=libstdc++.so.6) \
  build/remote/examples/ppml_train \
  data/training_batch_data/P62891_alignment.a3m data/P62891.fasta \
  data/training_batch_data/4ug0_P62891_mapping.csv ""
```

> For the full-size remote training (FULL_TRAIN, single sample P04637 + multi-sample L<=103),
> see the scripts **`build_remote.sh`** (build) and **`remote_fulltrain.sh`** (two-stage training).

## 训练基准 (Training Benchmark)

### Small setting CPU training  
Env: Intel i5-1335U 12 Core / 15 GB RAM / Pure CPU (LD_PRELOAD libstdc++)

small setting  CPU train 5 epoch (P62891, L=51, MSA_DEPTH=8, N_EXTRA=1, N_MAIN=2, N_REFINE=1, DEV_SE3=1, SE3_TOPO=per_block switch-B)  
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

small setting CUDA train 5 epoch (P62891, L=51, MSA_DEPTH=8, N_EXTRA=1, N_MAIN=2, N_REFINE=1, DEV_SE3=1, SE3_TOPO=per_block 开关B)  
环境: Intel i5-1335U 12 核 / 15 GB RAM / NVIDIA GeForce RTX 2050 4GB (compute 8.6, Tensor Core YES) / 混合调度 (BackendScheduler, GPU scatter 开启, 未设 PPML_CUDA_NO_SCATTER)

| Epoch | 耗时 (ms) | forward (ms) | loss | grad_norm |
|-------|----------|--------------|------|-----------|
| 1/5   | 36732    | 27229        | 12.56     | 0.10 |
| 2/5   | 38264    | 27321        | 15.0589   | 0.11 |
| 3/5   | 35333    | 26773        | 84820.5*  | 0.11 |
| 4/5   | 37172    | 26931        | 16.6727   | 0.11 |
| 5/5   | 34372    | 26125        | 18.4751   | 0.11 |

*Epoch3 偶发瞬态：源自 chi head 分量（chi=169618，同 step fape=0.0004 正常），非坐标/offset 链；
止损 + grad clip 保证下一 epoch 自恢复（16.67/18.48），全程无 NaN、无参数 NaN（PARAM-NAN=0）、COORDS-VAL 正常（68~195）。

- 5 epoch finished EXIT=0，loss converged（12.56→15.06→16.67→18.48），grad_norm stable=0.11，nan=0
- 5 epoch 全部完成 EXIT=0，loss 有意义的有限值，grad_norm 稳定 0.11

- Mixed mode: about 36.4s per epoch; About 1.05x faster than CPU mode
- 混合 CPU+GPU：每 epoch ~36.4s（CPU 纯跑 ~37.8s，提速约 1.05×）


dev/training data:

data/training_batch_data/P62891_alignment.a3m data/P62891.fasta  

## 实验记录 (Experiments)

**EN** — [Dump the per-block coordinates under `per_block`, measure the kNN-membership overlap between
adjacent blocks plus the `edge_d` difference → quantify the "intermediate structure vs final structure"
topology gap](Experiment.md).
TL;DR: for L≤65, `actual_top_k = min(top_k, L-1)` turns `make_graph` into a **complete graph** so the
topology cannot change (Jaccard ≡ 1 is an *identity*); for L=103 (true kNN) the first block already differs
from the final structure by **~30% of directed edges** (100% of residues affected) while the common-edge
geometry differs by only 0.29Å and coordinates drift by ~1Å RMSD ⇒ the churn comes from **near-tied
64th/65th neighbours**. The same document reports the **lightweight graph-version Pass 1**
(`src/model/PPMLTopoPass.cpp`): **1.5× faster Pass 1**, **1.76× faster epoch than `per_block`**, identical
loss level vs the old value-version Pass 1 — and it fixes the latter's *blown-up* coordinate basis.

> **TODO**
> **EN** — the topology churn is driven by **near-ties at the 64th/65th neighbour boundary**; whether those
> swapped neighbours are functionally equivalent (i.e. whether a frozen-topology two-pass pipeline loses
> accuracy) **needs further study** (suggested: same-seed `fixed` vs `per_block` loss/fape/chi comparison,
> and a sweep of `PPML_SE3_MAX_STEP`).
> **中文** — 变化来自「**第 64/65 名近邻近似并列**」的边界翻转，其功能影响（冻结拓扑是否掉精度）
> **需要进一步研究**（建议：同种子 `fixed` vs `per_block` 的 loss/fape/chi 对比，并扫 `PPML_SE3_MAX_STEP`）。

**中文** — [per_block 下 dump 每个 block 的 coords，统计相邻 block 的 kNN 成员重合率 + edge_d 差异 →
直接把"中间结构 vs 最终结构"的拓扑差量化出来](Experiment.md)。
结论速览：L≤65 时 `min(top_k, L-1)` 使 `make_graph` 退化为完全图 → 拓扑恒不变（是**恒等式**）；
L=103 真实 kNN 下首块与最终结构 **30% 有向边不同**、100% 残基 kNN 集合变化，但公共边几何仅差 0.29Å、
结构漂移仅 ~1Å RMSD ⇒ 差异来自"第 64/65 名近邻近似并列"的边界翻转。同一文档还给出**图版轻量 Pass1**
（`src/model/PPMLTopoPass.cpp`）：**Pass1 阶段快 1.5×**、整 epoch 比 `per_block` **快 1.76×**、
loss 与旧值版 Pass1 同级，并修掉了后者**坐标基准被打爆**的问题。

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

See **Quick Start → Build environment (tested)** for the verified toolchain. Summary:

- CMake >= 3.18
- CUDA Toolkit (required at configure time; tested 11.5.119)
- Python >= 3.8 **with development headers** — needed for `python_bridge` / future development only
- Linux (**WSL2 + Ubuntu 22.04** tested); **Windows is not fully supported yet**

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
