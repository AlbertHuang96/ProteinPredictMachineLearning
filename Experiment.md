# per_block 下 dump 每个 block 的 coords，统计相邻 block 的 kNN 成员重合率 + edge_d 差异 → 直接把"中间结构 vs 最终结构"的拓扑差量化出来

> **EN** — Dump the per-block coordinates under `PPML_SE3_TOPO=per_block`, measure the kNN-membership
> overlap between adjacent blocks and the `edge_d` difference, and thereby quantify the
> **"intermediate structure vs final structure" topology gap** (i.e. how much the topology a two-pass
> pipeline would freeze differs from the one `per_block` actually uses).
>
> 实验日期 / Date：**2026-09-12** ｜ 平台 / Platform：本机 WSL（Intel i5-1335U 12 核 / 15GB RAM / 纯 CPU）｜
> 配置 / Config：`PPML_SE3_TOPO=per_block`，`MSA=8`，8 个 block 边界（`extra=2 / main=4 / refine=2`），1 epoch

---

## 0. 结论速览 / TL;DR

> **EN** — **L≤65 is a degenerate case**: `actual_top_k = min(top_k, L-1)` (=50 for L=51) makes `make_graph`
> a **complete graph**, so the topology *cannot* change — Jaccard ≡ 1 is an **identity, not evidence of
> stability**; only `edge_d` (≤0.4Å) and coords (≤0.8Å) differ, hence negligible.
> **In the real regime (L=103, true kNN)** the coordinates drift by only ~1Å RMSD, yet up to **30% of
> directed edges are replaced** and **100% of residues** see their kNN set change; the geometry of the
> *common* edges differs by just 0.29Å ⇒ the churn comes from **near-ties at the 64th/65th neighbour
> boundary**, not from real structural change. Therefore a two-pass pipeline that freezes the final
> topology **does** produce a different graph for L≥66; its functional impact must be measured (see §4).
>
> **中文** — **L≤65 是退化情形**：`actual_top_k = min(top_k, L-1)`（L=51 时为 50）使 `make_graph` 变成
> **完全图**，拓扑**不可能**变化——"Jaccard ≡ 1"是**恒等式，而非稳定性证据**；此时只有 `edge_d`(≤0.4Å)
> 与 coords(≤0.8Å) 有差异，可忽略。
> **真实规模（L=103，真 kNN）**下坐标仅漂 ~1Å RMSD，却有**最多 30% 的有向边被替换**、**100% 残基**的
> kNN 集合发生变化，而**公共边**几何只差 0.29Å ⇒ 变化来自**第 64/65 名近邻的近似并列**，不是真实结构变化。
> 因此"冻结最终拓扑"的两遍式管线在 L≥66 时**确实产生不同的图**，其功能影响必须实测（见 §4）。

| | **L=51（P62891）** | **L=103（P62805）** |
|---|---|---|
| `make_graph` 选边 / edge selection | `min(64, L-1) = 50` → **完全图 / complete graph** | `64 < 102` → **真 kNN / true kNN**（保留 68%） |
| 有向边数 E / directed edges | 恒定 **2550** = L(L-1) | 7093~7107（10506 的 68%） |
| 相邻 block 边集 Jaccard / adjacent Jaccard | **1.0000（完全不变）** | **0.746**（首块）→ 稳态 **0.973~0.999** |
| 首块 vs 最终 Jaccard / first block vs final | 1.0000 | **0.7032**（新增 1231 / 消失 1243） |
| 邻居集合变化的残基 / residues with changed kNN set | **0%** | **100%**（首块）/ 稳态 93~100% |
| coords 累计漂移 / cumulative coords drift | **RMSD 0.81Å** | **RMSD 1.03Å** |
| 公共边 `\|Δedge_d\|` / common-edge geometry gap | 0.40Å（典型边长 18.9Å） | 0.29Å（典型边长 ~13Å） |

> **TODO**
> **EN** — the churn is driven by **near-ties at the 64th/65th neighbour boundary**; whether those swapped
> neighbours are *functionally* equivalent (and hence whether a frozen-topology two-pass pipeline loses
> accuracy) **needs further study**. Suggested measurements: `fixed` vs `per_block` loss/fape/chi at the same
> seed (bounds the effect), and a sweep of `PPML_SE3_MAX_STEP` (3Å → 10Å) to find the drift at which the
> neighbour set starts flipping.
> **中文** — 变化来自「**第 64/65 名近邻近似并列**」的边界翻转，其"换掉的邻居是否功能等价"（即冻结
> 拓扑的两遍式管线是否掉精度）**需要进一步研究**。建议实测：同种子下 `fixed` vs `per_block` 的
> loss/fape/chi（给出影响上界），并扫 `PPML_SE3_MAX_STEP`（3Å→10Å）找"开始换边"的漂移阈值。

![per_block 拓扑/坐标漂移对比 / topology & coord drift comparison](experiments/per_block_topo/per_block_topo_diff.png)

---

## 1. 实验方法 / Method

> **EN** — Instrumentation lives in `src/model/PPML.cpp` (`dump_se3_topo()` / `dump_coords()`, enabled by
> `PPML_DUMP_TOPO_DIR=<dir>`, zero cost when unset). Per block boundary it dumps (a) the topology actually
> used by `make_graph` — `int64[4]{B,L,E,0}` + `int64[2E] edge_index` + `float[3E] edge_d` +
> `float[B*L*9]` generating coords — and (b) the coords produced by `apply_coord_update()`.
> Analysis: `python_scripts/analyze_topo_dump.py`; plotting: `python_scripts/plot_topo_dump.py`;
> one-shot reproduction: `bash experiments/per_block_topo/run_dump.sh`.
>
> **中文** — 插桩在 `src/model/PPML.cpp`（`dump_se3_topo()` / `dump_coords()`，开关 `PPML_DUMP_TOPO_DIR`，
> 默认零开销）：每个 block 边界落盘 ①`make_graph` 实际使用的拓扑（`int64[4]{B,L,E,0}` + `int64[2E] edge_index`
> + `float[3E] edge_d` + `float[B*L*9]` 生成它的 coords）②`apply_coord_update()` 产出的坐标。
> 分析 `python_scripts/analyze_topo_dump.py`、绘图 `python_scripts/plot_topo_dump.py`、
> 一键复现 `bash experiments/per_block_topo/run_dump.sh`。

> ⚠️ **选边规则 / edge-selection rule**：`actual_top_k = min(top_k, L-1)`（`SE3Transformer.cpp:425`），
> 边 = **top-k CA 近邻 ∪ 序列邻（|i-j| < kmin=9）**，**有向**。L≤65 ⇒ `actual_top_k = L-1` ⇒ 完全图。

---

## 2. 原始指标 / Raw metrics

### L=51（P62891）— 完全图对照组，E 恒 2550 / complete-graph control

> **EN** — adjacent Jaccard ≡ 1.0000 (7/7 pairs), vs-final Jaccard ≡ 1.0000, 0 new/removed edges,
> **0/51 residues** changed, common-edge `|Δedge_d|` 0.007–0.204Å, coordinate step RMSD 0.03–0.73Å,
> cumulative 0.81Å.

| 指标（相邻 block，共 7 对）/ metric (adjacent, 7 pairs) | 值 / value |
|---|---|
| 有向边集 Jaccard / directed edge-set Jaccard | 1.0000 ×7 |
| vs **最终** block 的 Jaccard / vs FINAL | 1.0000 ×7 |
| 新增 / 消失边 / new / removed edges | 0 / 0 |
| 邻居集合变化的残基 / residues changed | **0/51 = 0%** |
| 公共边 `\|Δedge_d\|` 均值 / mean common-edge Δ | 0.007 ~ 0.204 Å |
| 边长变化（刚体不变）/ edge-length change (rigid-invariant) | 0.002 ~ 0.131 Å |
| coords 逐步 RMSD / per-step RMSD | 0.14 / 0.24 / 0.031 / 0.728 / 0.026 / 0.056 / 0.033 Å |
| coords 累计 RMSD / cumulative RMSD | 0.14 → 0.28 → 0.28 → 0.81 → 0.81 → 0.81 → **0.81 Å** |

### L=103（P62805）— 真 kNN，E≈7100（保留 68%）/ true kNN

> **EN** — first transition (block0→1) replaces ~1031 edges (Jaccard 0.746, 100% residues); steady state
> 2–3% per block (Jaccard 0.973–0.999); **block i vs FINAL** converges monotonically 0.703 → 0.998, i.e.
> the first block's topology differs from the final one by ~30% of edges. Common-edge geometry gap: 0.29Å.
> Cumulative coordinate drift: 1.03Å RMSD (Kabsch barely reduces it ⇒ non-rigid, but small).

| block 过渡 / transition | 0→1 | 1→2 | 2→3 | 3→4 | 4→5 | 5→6 | 6→7 |
|---|---|---|---|---|---|---|---|
| 相邻 Jaccard / adjacent | **0.746** | 0.975 | 0.999 | 0.976 | 0.974 | 0.973 | 0.998 |
| 相邻 新增/消失边 / adjacent new/removed | 1031/1033 | 92/88 | 3/5 | 85/90 | 91/95 | 97/100 | 9/9 |
| 相邻 变化残基% / adjacent residues changed | 100 | 94.2 | 1.0 | 93.2 | 99.0 | 100 | 6.8 |
| **block i vs 最终 / vs FINAL** Jaccard | **0.703** | 0.907 | 0.929 | 0.930 | 0.952 | 0.972 | 0.998 |
| vs 最终 新增/消失边 / vs FINAL new/removed | 1231/1243 | 342/352 | 255/269 | 253/265 | 172/179 | 98/101 | 9/9 |
| vs 最终 变化残基% / residues changed | 100 | 100 | 95.1 | 95.1 | 95.1 | 100 | 6.8 |
| vs 最终 `\|Δedge_d\|` 均值 / mean Δ | 0.29 | 0.235 | 0.226 | 0.178 | 0.142 | 0.138 | 0.092 Å |
| coords 逐步 RMSD / per-step | 0.049 | 0.408 | 0.296 | 0.006 | 0.512 | 0.575 | 0.186 Å |
| coords 累计 RMSD / cumulative | 0.049 | 0.411 | 0.506 | 0.506 | 0.718 | 1.009 | **1.026 Å** |

（`prev vs FINAL` 单调收敛 0.70 → 0.998 符合预期：越靠后的 block 越接近最终结构。/
Monotone convergence towards 1.0 as blocks approach the final structure.)

---

## 3. 判读 / Interpretation

> **EN** —
> 1. **Complete-graph artefact**: the L=51 invariance is an identity of `min(top_k, L-1)`; at that scale a
>    frozen-topology pipeline differs only in `edge_d` (≤0.4Å) and coords (≤0.8Å) ⇒ negligible.
> 2. **Real regime**: at L=103 the first block differs from the final structure by ~30% of edges (steady
>    state 2–3%), yet common-edge geometry differs by only 0.29Å and the structure drifts by only ~1Å RMSD
>    ⇒ the swapped neighbours are **near-ties**, i.e. the churn is a discretisation effect rather than a
>    real change of the interaction graph.
> 3. **Consequence for the two-pass (pass1 / lightweight) pipeline**: for L≥66 the frozen topology is
>    *not* identical to the per-block one (30% on the first block, 2–3% in steady state), so **"the geometry
>    barely moves" is not sufficient to conclude the effect is negligible** — measure it (§4).
>
> **中文** — ① 完全图恒等式（L=51 的"不变"不是稳定性证据）；② 真实规模下坐标只漂 ~1Å 却有 30% 边更替，
> 换掉的是近似并列的邻居（离散化效应，非真实相互作用变化）；③ 因此 L≥66 时冻结拓扑与逐块拓扑**不同**，
> 不能仅凭"几何差很小"判定影响可忽略 —— 需实测。

---

## 4. 建议的后续实测 / Suggested follow-up measurements

> **EN** — all of these use existing modes, no new code required.

| 实验 / experiment | 做法 / how | 给出什么 / what it yields |
|---|---|---|
| 冻结拓扑影响上界 / bound of frozen-topology effect | 同种子跑 `PPML_SE3_TOPO=fixed`（冻结**初始**拓扑）vs `per_block`，比 loss / fape / chi | "冻结拓扑"影响的**上界**（pass1/轻量版位于两者之间）/ upper bound; the two-pass pipeline lies in between |
| 换边阈值 / churn onset | 固定 `top_k`，扫 `PPML_SE3_MAX_STEP`（3Å → 10Å） | 多大坐标漂移开始显著换边（驱动量 = 漂移 / 第 k 名邻居间距）/ drift threshold |
| 替代规模验证 / smaller-scale proxy | 给 `make_graph` 加 `PPML_SE3_TOPK` 覆盖（默认 64），L=51 + `top_k=32`（保留率 64% ≈ L=103 的 63%） | 非完全图 regime 的低成本复现（本实验已完成，非必需）/ low-cost proxy (optional) |

---

## 5. 轻量版 Pass1 实现与对比 / Lightweight Pass1: implementation & comparison（2026-09-12）

> **EN** — The two-pass pipeline originally used the **value-version `PPMLModel::forward`** as Pass 1
> (~340 extra lines, with its own dropout masks → the value and graph tracks diverged by ~0.15).
> The lightweight version replaces it with a **graph-version topology pass** (`PPMLModel::topo_pass`,
> new file `src/model/PPMLTopoPass.cpp`): run the graph `forward_graph` once in `fixed`-topology mode
> (which internally materialises all SE3 offsets, chains `apply_coord_update`, and exposes the updated
> coordinates as `GraphOutput::coords`), and use those coordinates as Pass 2's `topo_coords`
> (`PPML_SE3_TOPO=pass1`, unchanged). Hence Pass 1 and Pass 2 share **the same graph ops and weights** —
> no duplicated implementation, no value/graph dropout mismatch. Knob: `PPML_TOPO_PASS=graph`,
> `PPML_TOPO_PASS_ITERS=n` (default 1).
>
> To make cross-run numbers comparable a reproducibility knob was added: **`PPML_SEED=<uint>`** seeds all
> weight initialisation and dropout (unset = previous behaviour). With `PPML_SEED=1`, four repeated runs
> give *identical* loss (14.54) — verified.

配置 / Config：P62891（L=51），MSA=8，extra/main/refine = 1/2/1，`PPML_DEV_SE3=1`，CPU，1 epoch，`PPML_SEED∈{1,2,3}`。

| 模式 / mode | Pass1 阶段 / phase | 总时长 / total | loss (s1/s2/s3) | 均值 / mean | NaN |
|---|---|---|---|---|---|
| `per_block`（开关B，现状 / current） | — | **34.1 s** | 17.05 / 16.21 / 14.48 | 15.91 | 0 |
| `fixed`（开关A） | — | 14.7 s | 14.54 / 16.73 / 15.20 | 15.49 | 0 |
| `pass1` + **值版** Pass1（旧 / old） | 6.40 s | 22.1 s | 14.10 / 15.14 / 15.12 | 14.79 | 0 |
| `pass1` + **图版** Pass1（新轻量版 / new） | **4.25 s** | **19.4 s** | 14.18 / 15.56 / 14.46 | **14.73** | 0 |
| `pass1` + 图版 Pass1，`ITERS=2` | 8.5 s | 23.1 s | 14.18 / 19.82 / 15.31 | 16.44 | 0 |

> **EN findings** —
> * **All five modes run clean: rc=0, `grad_norm=0.11`, no NaN, coordinates finite.**
> * **Speed**: the new Pass 1 phase is **1.5× faster** than the value Pass 1 (4.25s vs 6.40s) and the whole
>   epoch is **1.76× faster than `per_block`** (19.4s vs 34.1s).
> * **Accuracy (same seeds, hence comparable)**: the new graph Pass 1 matches the old value Pass 1
>   (mean loss 14.73 vs 14.79, per-seed deltas ±0.7 which is just the different dropout paths) — **no
>   regression**. (`per_block` averages 15.91 and `fixed` 15.49, but with only 3 seeds and a ±1.3 spread
>   this is *not yet* a significant ordering — more seeds needed.)
> * **Latent defect found & fixed by the new pass**: the **value Pass 1 blows up its coordinate basis**
>   (dumped CA mean 747Å / max 3995Å; typical edge length 1828Å vs 19Å for sane coords — consistent with the
>   code note "值版 0.03 ⇒ ~300Å/block 巨型扰动"). The new graph Pass 1 produces a **sane basis**
>   (CA 68.2–194.0Å = input range; per-block Δ ≈ 0.02Å), i.e. **Pass 2's SE3 now receives a sane geometry**
>   (`edge_d`) instead of a blown-up one.
> * `PPML_TOPO_PASS_ITERS=2` is *not* better here (mean 16.44, one outlier 19.82) — keep the default 1.

> **中文结论** — ① 五种模式全部 rc=0 / grad_norm 0.11 / **0 NaN**；② 新 Pass1 阶段**快 1.5×**（4.25s vs 6.40s），
> 整 epoch 比 `per_block` **快 1.76×**（19.4s vs 34.1s）；③ 同种子下新图版 Pass1 与旧值版 Pass1 **loss 同级**
> （均值 14.73 vs 14.79，逐种子差 ±0.7 属 dropout 路径差异）⇒ **无精度回退**（`per_block` 均值 15.91、
> `fixed` 15.49，但 3 个种子 ±1.3 的散布下**尚不能**判定优劣，需更多种子）；④ **顺带发现一个潜在缺陷**：
> **值版 Pass1 会把拓扑基准坐标打爆**（dump 显示 CA 均值 747Å / 最大 3995Å、典型边长 1828Å，而正常应为 ~19Å；
> 与代码注释"值版 0.03 ⇒ ~300Å/block 巨型扰动"一致），新图版 Pass1 的基准**正常**（CA 68.2–194.0Å，
> 每块仅变 ~0.02Å）⇒ Pass2 的 SE3 终于拿到**正常几何**（`edge_d`）；**该缺陷已修复，见 §6.1**。
> ⑤ `ITERS=2` 无收益（均值 16.44 且出现 19.82 的离群），保持默认 1。

复现 / reproduction:
```bash
bash experiments/topo_pass_compare/run_compare.sh                     # dev 配置，各模式 1 次
SEEDS="1 2 3" bash experiments/topo_pass_compare/run_compare.sh        # 固定种子（跨 run 可比）
bash experiments/topo_pass_compare/run_pass1_equiv.sh                  # 两实现拓扑基准对比（dump）
python3 python_scripts/compare_topo_dumps.py <dirA> <dirB>             # dump 等价性比较
python3 python_scripts/inspect_topo_dump.py <dir>                      # dump 量级自检
```

---

## 6. 值版坐标修复 + L=51 5-epoch 计时/内存 + L=103 内存标定（2026-09-12）

### 6.1 值版 Pass1 坐标爆炸：已修 / value Pass 1 blow-up: fixed

值版 `PPMLModel::forward` 的 **3 处 Step 4k**（`IterBlock::forward`、`FullBlock::forward`、`RefineBlock::forward`）
原来**直接加原始 SE3 offset**（既无 `×scale` 也无 clamp），而图版 `IterBlock::apply_coord_update` 早就是
`clamp(offset×0.03, ±3Å)` ⇒ 随机初始化下 offset 可达 O(1e3~1e8)，每个 block 推进数百~数千 Å，**把拓扑基准打爆**。
现已统一为同一规则（`PPML_SE3_MAX_STEP` 仍可覆盖，≤0 = 不 clamp）。

> **EN** — The three value-mode Step-4k coordinate updates (IterBlock/FullBlock/RefineBlock `forward`) applied the
> **raw SE3 offset** (no scale, no clamp), while the graph path `apply_coord_update` has always used
> `clamp(offset×0.03, ±3Å)`. With random init (|offset| up to 1e3~1e8) that pushed each block by hundreds of Å and
> blew up the topology basis. All three sites now use the same rule (override: `PPML_SE3_MAX_STEP`; ≤0 disables).

修复前后（L=51，`PPML_SE3_TOPO=pass1` 值版 Pass1，dump 统计 / dump statistics）：

| | `coords`\|CA\| mean / max | `edge_d` mean | 判读 / verdict |
|---|---|---|---|
| 修复前 / before | **747 / 3995 Å** | 1828 Å | 基准被打爆 / blown up |
| 修复后 / after | **140.9 / 193.9 Å** | **19.03 Å** | 正常（= 输入尺度）/ sane |

修复后该 run：`rc=0`、`Epoch 1/1 completed in 81523 ms`、loss 14.74、grad_norm 0.11。

### 6.2 L=51 轻量版 Pass1 5-epoch（时间 / 内存 / dump coords）

配置：P62891（L=51）、单样本、MSA=8、`extra2 + main4 + refine2`（8 个 block 边界，与 §2/§3 的拓扑对比一致）、
`PPML_SE3_TOPO=pass1 PPML_TOPO_PASS=graph`、CPU、`PPML_SEED=1`、5 epoch。

| epoch | wall / ms | forward / ms | Pass1 阶段 / ms | loss | grad_norm |
|---|---|---|---|---|---|
| 1 | 33163 | 7479 | 7193 | 15.640 | 0.11 |
| 2 | 32196 | 7162 | 7471 | 14.652 | 0.11 |
| 3 | 37537 | 8022 | 6855 | 15.802 | 0.11 |
| 4 | 31586 | 6312 | 6717 | 14.801 | 0.11 |
| 5 | 39731 | 8419 | 8410 | 19.817 | 0.11 |

- **总 wall 2:55.31**（5 epoch，均值 35.0 s/epoch，其中 Pass1 阶段 ~7.3 s、forward ~7.5 s）；
  `rc=0`、无 NaN、`grad_norm` 恒 0.11。
- **内存**：`/usr/bin/time -v` **Max RSS = 8.98 GB**；每个 epoch 的最终 compute（fwd+bwd）`[alloc-buffer]` 均为 **8.80 GB**。
- **dump**：80 个 `topo_*.bin` + 80 个 `coord_*.bin`（共 **6.0 MB**）；**epoch k 对应索引 `16(k-1) … 16k`**，
  其前 8 个是 Pass1（fixed）的边界、后 8 个是 Pass2 的边界。
- **coords 分析**（`python_scripts/analyze_L51_5ep.py`，输出 `L51_5ep_coords_analysis.txt`）：
  | 指标 / metric | 值 / value |
  |---|---|
  | CA 量级（Pass1 / Pass2 末边界） | 140.7~140.9 Å（稳定）/ 193.4~194.6 Å max |
  | `edge_d` 均值 | 18.92 ~ 19.12 Å（正常蛋白尺度） |
  | 相邻边界之间 CA 逐原子最大位移 | **恰好 3.000 Å（每个块都触到 ±3Å clamp 上限 ⇒ offset 幅度处于饱和区）** |
  | Pass1 末 vs Pass2 首（同 epoch）Kabsch RMSD | 0.06 ~ 0.71 Å（Pass2 确实从 Pass1 基准出发） |
  | epoch 间 Kabsch RMSD | 3.36 / 2.94 / 2.86 / 2.17 Å；**ep1→ep5 累计 3.28 Å**（最大原子位移 8.7 Å） |

> **EN** — 5 epochs of the lightweight two-pass pipeline on L=51 run clean (rc=0, no NaN, grad_norm 0.11) in
> **2:55 total (35 s/epoch)**, peak **Max RSS 8.98 GB**, dumping 160 files (6 MB). The coordinate dumps show a
> **stable, sane geometry** (CA 140.8 Å, edge 18.9–19.1 Å), the **±3 Å clamp saturating on every block**
> (offset magnitude in saturation regime), a Pass-2 start that matches the Pass-1 basis (RMSD 0.06–0.71 Å), and a
> **converging per-epoch evolution** (2.2–3.4 Å Kabsch RMSD/epoch, 3.28 Å over the whole run) rather than a drift.

### 6.3 L=103 内存标定：能不能在本机训练 / L=103 memory calibration

`GRAPH_DEBUG_GALLOCR=1`，单样本 P62805、MSA=8、CPU、1 epoch：

| 配置 / config | forward/reserve 阶段 | **最终 compute（fwd + bwd）** | 结果 / result |
|---|---|---|---|
| `pass1`(图版), extra2/main4/refine2（8 边界） | 2.76 GB（15655 节点） | **26.48 GB（39227 节点）** | rc=134 OOM |
| `per_block`, 同上 | 2.70 GB（14528 节点） | **26.50 GB（39232 节点）** | rc=134 OOM |
| `pass1`(图版), extra1/main1/refine1（3 边界） | 0.96 GB（5640 节点） | **9.47 GB（14749 节点）** | **rc=0，41.8 s/epoch ✓** |

- 26.5 GB 是**训练图本身**的开销：`big tensors(>1GB)=0`、`max tensor` 仅 0.03 GB（MSA 列注意力 scores）
  ⇒ 是 ~3.9 万个小节点同时存活（激活 + 梯度）的总量，**与 SE3 拓扑模式无关**（两模式 26.48 / 26.50 GB）。
- ⇒ 本机（15.4 GB RAM）**只能跑 ≤3~4 个 block 边界**的 L=103 训练（9.5 GB ✓ / 8 边界 26.5 GB ✗ 需服务器 ≥30 GB）。
- 与历史口径对照：早期"L=103 本地不行"针对的是**多样本（le103 两样本）+ MSA 64~512 + accum=4 + 16 边界**
  （19.5 ~ 65 GB，见 `repro_msa256.sh` 及 2026-09-11 记录）；本节给出的是**单样本 + MSA=8** 下的真实曲线。
- ⚠️ **更正**：§5 会话中"L=103 本地跑得动（峰值 2.92 GB）"是**中间 compute** 的峰值，训练在**最终 compute**
  阶段 OOM（当时未核对 `rc`）。该结论已作废；但已产出的 dump 数据**有效**（dump 发生在 OOM 之前的
  forward/reserve 阶段）⇒ §2/§3 的拓扑-漂移结论不受影响。

复现 / reproduction:
```bash
bash experiments/topo_pass_compare/run_L51_pass1_5ep.sh        # L=51 5 epoch + dump + 时间/内存汇总
bash experiments/topo_pass_compare/run_L103_pass1_3blk_5ep.sh  # L=103 3 边界 5 epoch（~10.6GB，本机可跑）
python3 python_scripts/analyze_L51_5ep.py <dump_dir> [n_epoch]  # coords/拓扑 dump 分析（泛化：自动推断边界数）
bash experiments/topo_pass_compare/step_verify_and_L103.sh    # 值版修复验证 + L=103（3 边界）
bash experiments/topo_pass_compare/diag_L103_mem.sh           # L=103 两模式 8 边界内存对照（会 OOM）
```

### 6.4 L=103（P62805）3 边界 5-epoch：时间 / 内存 / coords 分析

配置：单样本、MSA=8、`extra1 + main1 + refine1`（**3 个 block 边界** ⇒ 每 epoch 6 个 dump）、
`PPML_SE3_TOPO=pass1 PPML_TOPO_PASS=graph`、CPU、`PPML_SEED=1`、5 epoch。

| epoch | wall / ms | forward / ms | Pass1 阶段 / ms | loss | grad_norm |
|---|---|---|---|---|---|
| 1 | 46147 | 8987 | 7856 | 14.980 | 0.11 |
| 2 | 47763 | 11120 | 11015 | 15.431 | 0.11 |
| 3 | 48967 | 10886 | 10538 | 17.507 | 0.11 |
| 4 | 46741 | 10220 | 9798 | 16.208 | 0.11 |
| 5 | 41335 | 8319 | 8356 | 15.830 | 0.11 |

- 总 wall **3:51.89**（均值 46.4 s/epoch）；`rc=0`、无 NaN、`grad_norm` 恒 0.11。
- 内存：**Max RSS = 10.61 GB**；每 epoch 最终 compute `[alloc-buffer]` **10.66 GB**；forward/Pass1 阶段仅 **1.09 GB**
  （↔ 同 L=103 在 8 边界时最终 compute 26.5 GB ⇒ OOM，见 §6.3 ⇒ **边界数是本机可行性的决定变量**）。
- dump：30 `topo_*.bin` + 30 `coord_*.bin`（6.0 MB）；**epoch k = `coord` 索引 `[6(k-1), 6k)`**（前 3 个 = Pass1，后 3 个 = Pass2）。

**coords / 拓扑分析**（泛化脚本 `python_scripts/analyze_L51_5ep.py`，输出 `L103_3blk_5ep_coords_analysis.txt`）：

| 指标 / metric | 结果 / value |
|---|---|
| CA 量级（Pass1 / Pass2 末边界） | mean 4.0 Å，max 11.4~11.5 Å（全程稳定） |
| 相邻边界 CA 逐原子最大位移 | 多为 **3.000 Å（顶到 ±3Å clamp）**；少数 0.598 / 1.800 Å（未饱和） |
| `edge_d` 均值 | 0.85 ~ 0.91 Å（见下方 ⚠️ 输入结构说明） |
| **Pass1 内部各边界边界集 Jaccard** | **1.0000（3 个边界完全相同）⇒ "冻结基准" 机制按设计生效 ✓** |
| Pass2 内部各边界边界集 Jaccard | 1.0000（`pass1` 模式下 Pass2 用同一套 `topo_coords` 构图 ✓） |
| **Pass1 冻结基准 vs Pass2 基准（同 epoch）** | CA RMSD 0.018~0.450 Å；**边界集 Jaccard 0.816~0.836（约 18% 边不同）** |
| 边界集大小 E | 7102~7107（两端都固定条数 ⇒ Jaccard<1 是**成员变化**而非条数变化） |
| epoch 间（Pass2 末结构） | Kabsch RMSD 0.76~0.91 Å/epoch（**ep1→ep5 累计 0.93 Å**），最大原子位移 3.0~3.1 Å（= clamp 上界） |
| epoch 间边界集 Jaccard | 0.9265 / 0.9472 / 0.9487 / 0.9520；**ep1→ep5 0.9476**（拓扑跨 epoch 稳定 ~95%） |

> **EN** — L=103 runs locally only with a reduced block count (3 boundaries): 5 epochs in **3:52**
> (46 s/epoch), **Max RSS 10.6 GB**, clean (rc=0, no NaN). Coordinate analysis: per-block displacement
> saturates the ±3 Å clamp; **the frozen Pass-1 basis is exact (edge-set Jaccard 1.0000 across Pass-1's
> boundaries)**; the Pass-1 basis differs from Pass-2's by **~18 % of edges** (Jaccard 0.82–0.84) which is the
> real "frozen vs live topology" gap at L=103 (at L=51 it is an identity); across epochs the topology is
> **~95 % stable** (Jaccard 0.93–0.95) while the structure drifts only ~0.9 Å (Kabsch), i.e. **converging, not drifting**.

> ⚠️ **输入结构说明 / input-structure caveat**：P62805 的 `1zkk_P62805_mapping.csv` **只映射 24 个残基**
> ⇒ 103 个 CA 中多数坐标为退化值（`coords|CA| min=0.171`、`edge|d| min=0.000`、均值仅 0.85 Å、max 21.5 Å）。
> 该量级**不是本轮插桩/修复引入**：§2 的 `per_block/dumps/L103` 旧 dump 统计完全相同（mean 0.821~0.893）。
> ⇒ 读 L=103 的绝对几何指标（边长、半径）时要按"部分退化结构"解读；**边集/相对变化类指标不受影响**。
> **EN** — The L=103 sample's csv maps only 24 of 103 residues, so most CA coordinates are degenerate
> (mean edge 0.85 Å, `min=0.000`) — identical statistics in the older §2 dumps, i.e. a property of the input,
> not of the instrumentation. Absolute-geometry metrics for L=103 must be read with that in mind.

---

## 7. 产物清单 / Artifacts

> **EN** — all raw dumps are kept inside the repository (not `/tmp`) so they survive WSL restarts.

| 文件 / file | 说明 / description |
|---|---|
| `experiments/per_block_topo/run_dump.sh` | 一键复现拓扑 dump（两样本 × 8 block 边界）/ one-shot dump reproduction |
| `experiments/per_block_topo/dumps/{L51,L103}/` | 原始 dump（`topo_*.bin` / `coord_*.bin`，各 16 个）/ raw dumps |
| `experiments/per_block_topo/metrics.txt` | 完整指标（每对 block 的明细）/ full per-pair metrics |
| `experiments/per_block_topo/per_block_topo_diff.png` | 本页对比图（2×2）/ the figure above |
| `experiments/topo_pass_compare/run_compare.sh` | 五种模式对比（支持 `SEEDS` / `MODES` / `N_*`）/ mode comparison |
| `experiments/topo_pass_compare/run_pass1_equiv.sh` | 两个 Pass1 实现的拓扑基准对比（dump）/ Pass1 equivalence run |
| `experiments/topo_pass_compare/dumps/{pv,pg}/` | 值版 / 图版 Pass1 的 dump / Pass1 dumps |
| `experiments/topo_pass_compare/run_L51_pass1_5ep.sh` | L=51 轻量版 **5 epoch**（时间 / 内存 / dump，§6.2）/ timing & memory run |
| `experiments/topo_pass_compare/run_L103_pass1_3blk_5ep.sh` | **L=103 3 边界** 5 epoch（§6.4）/ L=103 3-boundary 5-epoch run |
| `experiments/topo_pass_compare/dumps/L103_pass1_3blk_5ep/` | 60 个 dump（每 epoch 6 个）/ L=103 5-epoch dumps |
| `experiments/topo_pass_compare/L103_3blk_5ep_coords_analysis.txt` | §6.4 的 coords / 拓扑分析输出 |
| `experiments/topo_pass_compare/dumps/L51_pass1_5ep/` | 5-epoch 的 160 个 dump（每 epoch 16 个，索引编码见 §6.2）/ 5-epoch dumps |
| `experiments/topo_pass_compare/L51_5ep_coords_analysis.txt` | §6.2 的 coords 分析输出 / coordinate analysis output |
| `experiments/topo_pass_compare/step_verify_and_L103.sh` | 值版坐标修复验证 + L=103（3 边界）§6.1/§6.3 |
| `experiments/topo_pass_compare/diag_L103_mem.sh` | L=103 两模式内存对照（§6.3）/ memory calibration |
| `python_scripts/analyze_L51_5ep.py` | 5-epoch coords 分析（clamp / 漂移 / epoch 演化）/ analysis script |
| `python_scripts/analyze_topo_dump.py` / `plot_topo_dump.py` / `compare_topo_dumps.py` / `inspect_topo_dump.py` | 分析 / 绘图 / 比较 / 自检脚本 / analysis & plotting |
| `src/model/PPML.cpp`（`dump_se3_topo` / `dump_coords`） | 插桩（`PPML_DUMP_TOPO_DIR`，默认关）/ instrumentation |
| `src/model/PPMLTopoPass.cpp`、`include/ppml/Model.h`、`examples/train.cpp`、`include/ppml/Core.h` | 轻量版 Pass1 + `PPML_SEED` 可复现种子 / lightweight Pass 1 + reproducible seed |

> 训练侧元信息 / run metadata：L=51 的 dump 轮 `Epoch 1/1 completed, loss≈13.8, grad_norm 0.11`
> （与同配置常规回归一致 ⇒ 插桩无副作用 / instrumentation is side-effect free）。
