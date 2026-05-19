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


# process each template pair feature
        templ = self.templ_stack(templ, rbf_feat, t1d, use_checkpoint=use_checkpoint, p2p_crop=p2p_crop, is_prot=is_prot) # (B, T, L,L, d_templ)
        # symmetry and pseudocycle diffusion for singular/non-repeat applications does not need to process is_sm
        # but include it for generality and future-proofing

        def get_res_atom_dist(idx, bond_feats, dist_matrix, sm_mask, minpos_res=-32, maxpos_res=32, maxpos_atom=8, cyclize=None):
    '''
    Calculates residue and atom bond distances of protein/SM complex. Used for positional
    embedding and structure module. 2nd version (2022-9-19); handles atomized proteins.
 
    Input:
        - idx: residue index (B, L)
        - bond_feats: bond features (B, L, L)
        - dist_matrix: precomputed bond distances (B, L, L) NOTE: need to run nan_to_num to remove infinities
        - sm_mask: boolean feature (L). True if a position represents atom, False otherwise
        - minpos_res: minimum value of residue distances
        - maxpos_res: maximum value of residue distances
        - maxpos_atom: maximum value of atom bond distances
 
    Output:
        - res_dist: residue distance (B, L, L)
        - atom_dist: atom bond distance (B, L, L)
    '''
    bond_feats = bond_feats[0] # assume batch = 1
    L = bond_feats.shape[0]
    device = bond_feats.device

    prot_mask_2d = (~sm_mask[None,:]) * (~sm_mask[:,None])
    inter_mask_2d = (~sm_mask[None,:]) * (sm_mask[:,None]) + (sm_mask[None,:]) * (~sm_mask[:,None])
 
    seqsep = idx[0,None,:] - idx[0,:,None] # (L, L)
    if cyclize is not None:
        mask = cyclize[:,None]*cyclize[None,:]
        ncyc = torch.sum(cyclize)
        seqsep[mask*(seqsep>ncyc//2)] -= ncyc
        seqsep[mask*(seqsep<-ncyc//2)] += ncyc
 
    res_dist_prot = torch.clamp(seqsep, min=minpos_res, max=maxpos_res) # (L, L) intra-protein
    res_dist_sm = torch.full((L,L), maxpos_res+1, device=device) # (L, L) with "unknown" res. dist. token
 
    # small molecule atom bond graph
    atom_dist_sm = torch.nan_to_num(dist_matrix, posinf=maxpos_atom)[0].long() # this comes through the dataloader so it is batched
    atom_dist_prot = torch.full((L,L), maxpos_atom+1, device=device)
 
    #fd new impl
    i_s, j_s = torch.where(bond_feats==6)
    i_sm = i_s[sm_mask[i_s]]
    i_prot = j_s[sm_mask[i_s]]
    res_dist_inter = torch.full((L,L), maxpos_res, device=device)
    atom_dist_inter = torch.full((L,L), maxpos_atom, device=device)
    if i_prot.shape[0] > 0:
        closest_prot_res = i_prot[torch.argmin(atom_dist_sm[sm_mask,:][:,i_sm], dim=-1)]
        res_dist_inter[sm_mask,:] = res_dist_prot[closest_prot_res,:]
        res_dist_inter[:,sm_mask] = res_dist_prot[:,closest_prot_res]
 
        closest_atom = i_sm[torch.argmin(torch.abs(res_dist_prot[~sm_mask,:][:,i_prot]), dim=-1)]
        atom_dist_inter[~sm_mask,:] = atom_dist_sm[closest_atom,:] + 1
        atom_dist_inter[:,~sm_mask] = atom_dist_sm[:,closest_atom] + 1
    
    res_dist = res_dist_prot * prot_mask_2d + res_dist_inter * inter_mask_2d + res_dist_sm * sm_mask_2d
    atom_dist = atom_dist_prot * prot_mask_2d + atom_dist_inter * inter_mask_2d + atom_dist_sm * sm_mask_2d