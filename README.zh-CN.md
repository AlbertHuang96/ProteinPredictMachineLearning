
# ProteinPredictMachineLearning 项目
PPML 项目  
受 RosettaFoldAllAtom(RFAA) 与 GGML 启发  

英文版 / English version: [README.md](README.md)

## 引用项目

本仓库是独立的 C++ 重实现（研究 / 工程原型）；下面两个项目是它的概念与工程参考。

| # | 项目 | 参考点 |
|---|---|---|
| 1 | RoseTTAFold-All-Atom (RFAA) — Baker lab<br><https://github.com/baker-laboratory/RoseTTAFold-All-Atom> | 全原子结构预测；三轨（1D/2D/3D）网络 + SE(3) 等变注意力；本项目对齐其训练管线与配置。 |
| 2 | ggml — 张量库与推理运行时<br><https://github.com/ggml-org/ggml> | 无依赖的 C/C++ 张量运行时（内存池 + 图执行器），本项目 C++ 运行时的工程参照。 |

## 训练基准

### Small setting CPU 训练
环境: Intel i5-1335U 12 核 / 15 GB RAM / 纯 CPU (LD_PRELOAD 系统 libstdc++)

small setting CPU 训练 5 epoch（P62891, L=51, MSA_DEPTH=8, N_EXTRA=1, N_MAIN=2, N_REFINE=1, DEV_SE3=1, SE3_TOPO=per_block 开关B）  

| Epoch | 耗时 (ms) | forward (ms) | loss | grad_norm |
|-------|----------|--------------|------|-----------|
| 1/5   | 34642    | 25576        | 14.16    | 0.11 |
| 2/5   | 35859    | 25254        | 15.1777  | 0.11 |
| 3/5   | 43021    | 28667        | 14.7508  | 0.11 |
| 4/5   | 38636    | 27085        | 14.3692  | 0.11 |
| 5/5   | 37338    | 25205        | 14.3877  | 0.11 |

- 5 epoch 全部完成 EXIT=0，loss 全有限（无 NaN），grad_norm 稳定 0.11


### Small setting 混合训练 5 epoch
环境: Intel i5-1335U 12 核 / 15 GB RAM / NVIDIA GeForce RTX 2050 4GB (compute 8.6, Tensor Core YES) / 混合调度 (BackendScheduler, GPU scatter 开启, 未设 PPML_CUDA_NO_SCATTER)

small setting CUDA 训练 5 epoch（P62891, L=51, MSA_DEPTH=8, N_EXTRA=1, N_MAIN=2, N_REFINE=1, DEV_SE3=1, SE3_TOPO=per_block 开关B）  

| Epoch | 耗时 (ms) | forward (ms) | loss | grad_norm |
|-------|----------|--------------|------|-----------|
| 1/5   | 36732    | 27229        | 12.56     | 0.10 |
| 2/5   | 38264    | 27321        | 15.0589   | 0.11 |
| 3/5   | 35333    | 26773        | 84820.5\* | 0.11 |
| 4/5   | 37172    | 26931        | 16.6727   | 0.11 |
| 5/5   | 34372    | 26125        | 18.4751   | 0.11 |

*Epoch3 偶发瞬态：源自 chi head 分量（chi=169618，同 step fape=0.0004 正常），非坐标/offset 链；
止损 + grad clip 保证下一 epoch 自恢复（16.67/18.48），全程无 NaN、无参数 NaN（PARAM-NAN=0）、COORDS-VAL 正常（68~195）。

- 5 epoch 全部完成 EXIT=0，loss 有意义的有限值，grad_norm 稳定 0.11

- 混合 CPU+GPU：每 epoch ~36.4s（CPU 纯跑 ~37.8s，提速约 1.05×）


dev/训练数据：

data/training_batch_data/P62891_alignment.a3m data/P62891.fasta  

### 远程 FULL_TRAIN 多样本基准（le103，块 2-4-2，10 epoch）— 2026-09-21

环境: AMD Ryzen Threadripper PRO 3955WX 32 线程 / 220 GB RAM / NVIDIA A800 80GB / 混合调度 (BackendScheduler, CPU+CUDA) + staging async

配置: `FULL_TRAIN=1 SKIP_P04637=1` / 数据集 `training_batch_data_le103` / MSA_DEPTH=256 / 块 `extra 2 + main 4 + refine 2`（8 个边界）/ accum=4 / 10 epoch / lr=1e-4 / clip=0.1 / `SE3_TOPO=per_block`（可学习缩放）/ `PPML_USE_CUDA=1` / `PPML_STAGING_ASYNC=1` / 显存预算自动（4/5 空闲）  

样本: 2 个（`P62805` L=103 / `P62891` L=51）；共 20 个样本步（10 epoch × 2）；5 次优化步（accum=4）  

| 指标 | 值 |
|---|---|
| 总耗时 | 23,264.1 s（6 小时 27 分）|
| 每 epoch（2 样本）| ≈2,326 s（38.8 分）|
| 每样本步 | ≈1,163 s（19.4 分）|
| 优化步 | 5 |
| avg loss（20 样本步）| 12.5463 |
| grad_norm | 0.1000（clip 饱和）；其中 1 次 0.0000 |
| 峰值（gallocr）| CUDA 18.7 GB / CPU 6.2 GB |
| 退出 | 0 ✓ 10/10 epoch 完成 |

| Epoch | 耗时 s（估算）\* | P62805 (L=103) loss | P62891 (L=51) loss | step grad_norm |
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

\* 该多样本路径不打印 per-epoch ms ⇒ 每 epoch 耗时按"日志行跨度 × 总耗时/总行数"线性估算（估算合计 23,483 s vs 实测 23,264 s，偏差约 1%）；总耗时与每 epoch 均值是实测值。

- 10 epoch 全部完成 EXIT=0，无 NaN（`FWD-NAN=0`），loss 全有限，`[ctx]` 对象数持平（无累积）。  



## 实验记录

[per_block 下 dump 每个 block 的 coords，统计相邻 block 的 kNN 成员重合率 + edge_d 差异 →
直接把"中间结构 vs 最终结构"的拓扑差量化出来](Experiment.md)。
结论速览：L≤65 时 `min(top_k, L-1)` 使 `make_graph` 退化为完全图 → 拓扑恒不变（是恒等式）；
L=103 真实 kNN 下首块与最终结构 30% 有向边不同、100% 残基 kNN 集合变化，但公共边几何仅差 0.29Å、
结构漂移仅 ~1Å RMSD ⇒ 差异来自"第 64/65 名近邻近似并列"的边界翻转。同一文档还给出图版轻量 Pass1
（`src/model/PPMLTopoPass.cpp`）：Pass1 阶段快 1.5×、整 epoch 比 `per_block` 快 1.76×、
loss 与旧值版 Pass1 同级，并修掉了后者坐标基准被打爆的问题。

> TODO
> 变化来自「第 64/65 名近邻近似并列」的边界翻转，其功能影响（冻结拓扑是否掉精度）
> 需要进一步研究（建议：同种子 `fixed` vs `per_block` 的 loss/fape/chi 对比，并扫 `PPML_SE3_MAX_STEP`）。

## TODO

- residx 目前是理想化连续索引 0..L-1，未用 CSV/PDB 的真实 ResNum；单链连续场景够用，但多链/缺残基(gap)/非标准编号时会丢失真实序列间隔信息（gap 应拉开但当前视为相邻）。CSV 已有 PDB_ResNum 列，尚未用于构建 residx。
- 模板 pair 注入 T>1（待实现）：`PPMLModel::forward_graph` 的模板注入中，state 分支已用全部 T 模板作 cross-attn key；但 pair 分支当前仅用 t=0 单模板（值版 `PairTrack::inject_template` 亦是 T=1 语义）。原因：图基础设施 4D 上限（`kernel_concat`/`kernel_permute` 只支持 ≤4D），且缺沿 dims[3] 的归约 op（`sum`/`mean` 只做全归约，`sum_rows` 只沿 dims[1]）。完整方案：① 新增沿任意维的 reduce op；或 ② 把 templ_pair `[64,L,L,T]` 经 permute 重排为 `[64,1,1,L*L*T]` 作为多模板 kv（T 折叠进 key 长度），query 仍为 `B*L*L`。需同步值版语义。

### 扭转角索引参考
- Negative index indicates the previous residue（负索引表示上一个残基）
- 顺序：
  - omega/phi/psi: 0-2
  - chi_1-4 (prot): 3-6
  - cb/cg bend: 7-9
  - eps(p)/zeta(p): 10-11
  - alpha/beta/gamma/delta: 12-15
  - nu2/nu1/nu0: 16-18
  - chi_1 (na): 19
