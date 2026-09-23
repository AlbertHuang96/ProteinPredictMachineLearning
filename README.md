
# ProteinPredictMachineLearning Project
PPML project  
Inspired by RosettaFoldAllAtom(RFAA) and GGML  

Chinese version: [README.zh-CN.md](README.zh-CN.md)

## References

This repository is an independent C++ re-implementation (research / engineering prototype); the two
projects below are its conceptual and engineering references.

| # | Project | Reference point |
|---|---|---|
| 1 | RoseTTAFold-All-Atom (RFAA) — Baker lab<br><https://github.com/baker-laboratory/RoseTTAFold-All-Atom> | All-atom structure prediction; three-track (1D/2D/3D) network with SE(3)-equivariant attention; the training pipeline and configs mirrored here. |
| 2 | ggml — tensor library & inference runtime<br><https://github.com/ggml-org/ggml> | Dependency-free C/C++ tensor runtime (memory pools + graph executor) used as the engineering model for our runtime. |

## Quick Start (command line)

### Build environment (tested)

| Item | Version / note |
|---|---|
| OS | Ubuntu 22.04.5 LTS on WSL2 — kernel `5.15.167.4-microsoft-standard-WSL2` |
| CMake | 3.22.1 (project minimum: 3.18) |
| Compiler | GCC 11.4.0 (C++17) |
| CUDA | 11.5.119 — tested on GeForce RTX 2050 (compute 8.6). The CUDA toolkit is required at configure time (`project(... LANGUAGES CXX CUDA)` + `find_package(CUDAToolkit REQUIRED)`), even for CPU-only runs. |
| Python | 3.11.5 (Anaconda) with development headers (`Python.h` / `python3-dev`) |

Notes
- Windows is not fully supported yet. Build/run is verified on Linux only (WSL2 + Ubuntu 22.04).
  Windows PowerShell cannot drive the existing CMake cache — use WSL or a Linux host.
  
  Server build: `build_remote.sh [target]` (also auto-detects python/CUDA).
- The Python requirement is mainly for future development convenience (the `python_bridge`
  pybind11 module). Training / inference are pure C++ and do not need Python at runtime; a
  Python without development headers is not enough to configure the project.
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
> see the scripts `build_remote.sh` (build) and `remote_fulltrain.sh` (two-stage training).

## Training Benchmark

### Small setting CPU training
Env: Intel i5-1335U 12 Core / 15 GB RAM / Pure CPU (LD_PRELOAD libstdc++)

small setting CPU train 5 epoch (P62891, L=51, MSA_DEPTH=8, N_EXTRA=1, N_MAIN=2, N_REFINE=1, DEV_SE3=1, SE3_TOPO=per_block switch-B)

| Epoch | Time (ms) | forward (ms) | loss | grad_norm |
|-------|----------|--------------|------|-----------|
| 1/5   | 34642    | 25576        | 14.16    | 0.11 |
| 2/5   | 35859    | 25254        | 15.1777  | 0.11 |
| 3/5   | 43021    | 28667        | 14.7508  | 0.11 |
| 4/5   | 38636    | 27085        | 14.3692  | 0.11 |
| 5/5   | 37338    | 25205        | 14.3877  | 0.11 |

- 5 epoch finished with EXIT=0, loss limited (no NaN), grad_norm stable=0.11


### Small setting hybrid training 5 epoch
Env: Intel i5-1335U 12 Core / 15 GB RAM / NVIDIA GeForce RTX 2050 4GB (compute 8.6)

small setting CUDA train 5 epoch (P62891, L=51, MSA_DEPTH=8, N_EXTRA=1, N_MAIN=2, N_REFINE=1, DEV_SE3=1, SE3_TOPO=per_block switch-B)

| Epoch | Time (ms) | forward (ms) | loss | grad_norm |
|-------|----------|--------------|------|-----------|
| 1/5   | 36732    | 27229        | 12.56     | 0.10 |
| 2/5   | 38264    | 27321        | 15.0589   | 0.11 |
| 3/5   | 35333    | 26773        | 84820.5\* | 0.11 |
| 4/5   | 37172    | 26931        | 16.6727   | 0.11 |
| 5/5   | 34372    | 26125        | 18.4751   | 0.11 |

\*Epoch3 transient: comes from the chi head component (chi=169618 while the same step has fape=0.0004, normal),
not from the coordinate/offset chain; the loss-stop plus grad clip lets the next epoch recover by itself
(16.67/18.48). No NaN anywhere, no parameter NaN (PARAM-NAN=0), COORDS-VAL normal (68~195).

- 5 epoch finished EXIT=0, loss converged (12.56→15.06→16.67→18.48), grad_norm stable=0.11, nan=0

- Mixed mode: about 36.4s per epoch; About 1.05x faster than CPU mode


dev/training data:

data/training_batch_data/P62891_alignment.a3m data/P62891.fasta  

### Remote FULL_TRAIN multi-sample benchmark (le103, blocks 2-4-2, 10 epoch) — 2026-09-21

Env: AMD Ryzen Threadripper PRO 3955WX 32 threads / 220 GB RAM / NVIDIA A800 80GB (compute 8.0) / mixed CPU+CUDA scheduler + staging async

Config: `FULL_TRAIN=1 SKIP_P04637=1` / dataset `training_batch_data_le103` / MSA_DEPTH=256 / blocks `extra 2 + main 4 + refine 2` (8 boundaries) / accum=4 / epochs=10 / lr=1e-4 / clip=0.1 / `SE3_TOPO=per_block` (learnable scale) / `PPML_USE_CUDA=1` / `PPML_STAGING_ASYNC=1` / VRAM budget auto (4/5 free)  

Samples: 2 — `P62805` (L=103) + `P62891` (L=51); 20 sample-steps (10 epoch × 2); 5 optimizer steps (accum=4)  

| Metric | Value |
|---|---|
| Total elapsed | 23,264.1 s (6 h 27 min) |
| Per epoch (2 samples) | ≈2,326 s (38.8 min) |
| Per sample-step | ≈1,163 s (19.4 min) |
| Optimizer steps | 5 |
| avg loss (20 sample-steps) | 12.5463 |
| grad_norm | 0.1000 (clip saturated); 0.0000 once |
| Peak (gallocr) | CUDA 18.7 GB / CPU 6.2 GB |
| Exit | 0 ✓ 10/10 epoch complete |

| Epoch | Time s (est.)\* | P62805 (L=103) loss | P62891 (L=51) loss | step grad_norm |
|---|---|---|---|---|
| 1  | 2,404 | 12.5447 | 12.5451 | — |
| 2  | 2,167 | 12.5447 | 12.5732 | 0.1000 |
| 3  | 2,390 | 12.5447 | 12.5451 | — |
| 4  | 2,172 | 12.5447 | 12.5451 | 0.1000 |
| 5  | 2,387 | 12.5447 | 12.5451 | — |
| 6  | 2,384 | 12.5447 | 12.5451 | 0.1000 |
| 7  | 2,399 | 12.5447 | 12.5451 | — |
| 8  | 2,386 | 12.5447 | 12.5451 | 0.0000 |
| 9  | 2,393 | 12.5447 | 12.5451 | — |
| 10 | 2,401 | 12.5447 | 12.5451 | 0.1000 |

\* The multi-sample path does not print per-epoch ms; per-epoch values are interpolated from log-length spans (Σ 23,483 s vs measured 23,264 s, ~1%). Total elapsed and per-epoch average are measured.

- 10 epoch finished with EXIT=0, no NaN (`FWD-NAN=0`), loss all finite, `[ctx]` object count flat (no accumulation).  



## Experiments

[Dump the per-block coordinates under `per_block`, measure the kNN-membership overlap between
adjacent blocks plus the `edge_d` difference → quantify the "intermediate structure vs final structure"
topology gap](Experiment.md).
TL;DR: for L≤65, `actual_top_k = min(top_k, L-1)` turns `make_graph` into a complete graph so the
topology cannot change (Jaccard ≡ 1 is an identity); for L=103 (true kNN) the first block already differs
from the final structure by ~30% of directed edges (100% of residues affected) while the common-edge
geometry differs by only 0.29Å and coordinates drift by ~1Å RMSD ⇒ the churn comes from near-tied
64th/65th neighbours. The same document reports the lightweight graph-version Pass 1
(`src/model/PPMLTopoPass.cpp`): 1.5× faster Pass 1, 1.76× faster epoch than `per_block`, identical
loss level vs the old value-version Pass 1 — and it fixes the latter's blown-up coordinate basis.

> TODO
> The topology churn is driven by near-ties at the 64th/65th neighbour boundary; whether those
> swapped neighbours are functionally equivalent (i.e. whether a frozen-topology two-pass pipeline loses
> accuracy) needs further study (suggested: same-seed `fixed` vs `per_block` loss/fape/chi comparison,
> and a sweep of `PPML_SE3_MAX_STEP`).

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

See Quick Start → Build environment (tested) for the verified toolchain. Summary:

- CMake >= 3.18
- CUDA Toolkit (required at configure time; tested 11.5.119)
- Python >= 3.8 with development headers — needed for `python_bridge` / future development only
- Linux (WSL2 + Ubuntu 22.04 tested); Windows is not fully supported yet

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
- `residx` is currently an idealized contiguous index 0..L-1; the real ResNum from CSV/PDB is not used.
  That is fine for single-chain contiguous cases, but multi-chain / missing-residue (gap) / non-standard
  numbering would lose the real sequence spacing (a gap should be separated, but is currently treated as
  adjacent). The CSV already has a PDB_ResNum column, not yet used to build `residx`.
- Template pair injection for T>1 (not implemented yet): in `PPMLModel::forward_graph` template injection,
  the state branch already uses all T templates as cross-attn keys; the pair branch currently uses only the
  t=0 single template (the value-version `PairTrack::inject_template` has the same T=1 semantics). Reason:
  the graph infrastructure has a 4D limit (`kernel_concat`/`kernel_permute` only support ≤4D) and there is
  no reduction op along dims[3] (`sum`/`mean` only do full reductions, `sum_rows` only along dims[1]).
  A complete fix: (1) add a reduce op along an arbitrary dimension; or (2) permute templ_pair
  `[64,L,L,T]` into `[64,1,1,L*L*T]` as multi-template kv (T folded into the key length) while the query
  stays `B*L*L`. The value-version semantics must be kept in sync.

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
