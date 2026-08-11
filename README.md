
# ProteinPredictMachineLearning Project
PPML project  
Inspired by RosettaFoldAllAtom  
and   
llama.cpp and GGML  

## Project Structure

Main goal:  

predict the 3D structure of a protein with limited compute resources,   
i.e. personal computer  
still leave potential to running on a GPU server  
or multiplatform deploy  
Theoretically, it could predict structure of all-atom protein.  

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
