#!/usr/bin/env bash
# ============================================================
# RFAA-Cpp remote server FULL_TRAIN training script
# Usage: bash remote_fulltrain.sh
# Overview:
#   1. Stage 1: single sample P04637 (with templates P04637_template_coords/ + P04637_pdbs/ true coords)
#      [skipped by default] SKIP_P04637=1 (2026-08-28: high memory peak, skipped for now; set SKIP_P04637=0 to run)
#   2. Stage 2: multi-sample (PPML_MULTI_SAMPLE=1, dataset default = training_batch_data_le103,
#      contains only L<=103 samples: P62805(L=103) + P62891(L=51), keeps the pair-track memory peak bounded)
#   3. Log: remote_fulltrain_YYYYMMDD_HHMMSS.log
# Env overrides: FULL_TRAIN(default true) PPML_NUM_EPOCHS(default 10) PPML_LR(default 1e-4)
#                PPML_CLIP_NORM(default 0.1) PPML_ACCUM(default 4) PPML_MSA_DEPTH(global default 512)
#                PPML_MULTI_MSA_DEPTH(stage 2 MSA depth, **default 256**; stage 1 fixed 256)
#                PPML_CUDA_NO_SCATTER(default 1, scatter falls back to CPU for reliable mixed training)
#                SKIP_P04637(default 1) PPML_DATASET_DIR(default training_batch_data_le103)
#   【2026-09-20 新增】
#                PPML_STAGING_ASYNC(default 1)     pinned 双缓冲异步 H2D/D2H（=0 关闭 ✓）
#                PPML_GPU_BUDGET_MB(默认空 = 空闲显存×4/5)   显存预算
#                GRAPH_DEBUG_SCHED(default 1)      **splits 汇总**（split_graph/PLAN/COMPUTE/backend[]/budget 逐出）
#                GRAPH_DEBUG_GALLOCR(default 1)    [gallocr] reserve done（各后端峰值）
#                PPML_DEBUG_ALLOC(default 1)       [alloc-buffer] 每次分配（=0 则只自动打 ≥1GB 的 ✓）
#                GRAPH_DEBUG_SCHED_VERBOSE(default 0) 逐**节点** dump（很吵 ⚠️，排障才开）
#                GRAPH_DEBUG_STAGING(default 0)    逐条 staging 拷贝（很吵 ⚠️）
#                RFAA_LEAN=1                       一键精简：关掉上述全部调试打印 ✓
#                （topo pass 如需要：PPML_DEV_SE3=1 PPML_SE3_TOPO=pass1 PPML_TOPO_PASS=graph）
#   【2026-09-21 新增】多样本内存自检（方案 B：grad 张量化 ⇒ 每样本释放 ✓）
#                PPML_DEBUG_CTX(default 1)         每样本打印 '[ctx] … n_objects=…'（应持平 ✓；RFAA_LEAN=1 关闭）
#                · 跑完会自动做两件自检：① 是否有 '[grad-carrier]'（没有 ⇒ 服务器二进制是旧版 ⚠️ 需重编）；
#                  ② '[ctx] n_objects' 是否持平（单调增长 ✗ ⇒ accum 窗口内样本图仍在累积 ⚠️）。
#                · 旧版症状：`ContextImpl.cpp:178 Assertion "Context memory exhausted"` ⇒ exit 134（SIGABRT）✗
#                · 兜底开关（万一仍累积）：PPML_ACCUM=1（或 2）+ PPML_MULTI_MSA_DEPTH=128 ✓
# Progress: neither stage has a timeout, training ends naturally. Live progress:
#   tail -f remote_fulltrain_*.log  and watch 'Epoch N/M completed ... loss:' lines;
#   if no output for a long time, check whether '[FWD-GRAPH] ... build ...' was printed (slow first full-size build is normal).
# ============================================================
set -u

# ---------- 0. Server environment paths ----------
# Project root: default is the directory of this script (project root), override via RFAA_PROJ
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${RFAA_PROJ:-$SCRIPT_DIR}"
# Executable (built by build_remote.sh, output under build/remote/)
BIN="${RFAA_BIN:-${PROJ}/build/remote/examples/ppml_train}"
# Python lib dir (server conda/venv lib, provides libpython3.x.so)
PYLIB="${RFAA_PYLIB:-/root/anaconda3/lib}"
# System gcc libstdc++ (avoid conda's older libstdc++ missing GLIBCXX_3.4.30)
# On the server, confirm with `ls /usr/lib/x86_64-linux-gnu/libstdc++.so.6`; set empty if the conda one is new enough
STDCPP_PRELOAD="${RFAA_STDCPP:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

# ---------- Training hyper-parameters (overridable via env) ----------
FULL_TRAIN="${FULL_TRAIN:-true}"
NUM_EPOCHS="${PPML_NUM_EPOCHS:-10}"
LR="${PPML_LR:-1e-4}"
CLIP="${PPML_CLIP_NORM:-0.1}"
ACCUM="${PPML_ACCUM:-4}"
MSA_DEPTH="${PPML_MSA_DEPTH:-512}"
# Stage 2 MSA depth: lowered 512 -> 256 on 2026-09-11.
#   Rationale (local peak calibration peak_CPU_GB ~= 12.9 + 0.102*N, le103 / accum=4 / CPU single backend):
#     N=512 ~= 65GB > server available 55.8GB (OOM'd last round, exit 137); N=256 ~= 39GB (actual alloc 43.9GB) fits.
#   Restore 512: PPML_MULTI_MSA_DEPTH=512 bash remote_fulltrain.sh (confirm server free memory first, or lower PPML_ACCUM).
MULTI_MSA_DEPTH="${PPML_MULTI_MSA_DEPTH:-256}"
USE_CUDA="${PPML_USE_CUDA:-1}"
CUDA_SCHED="${PPML_CUDA_SCHED:-1}"
#  Mixed training reliability: force OP_SCATTER_ADD to fall back to CPU (GPU scatter cross-split race not fixed yet;
#    observed loss blow-up / launch failure on GPU scatter, loss 15.47 normal after CPU fallback).
#    Set PPML_CUDA_NO_SCATTER=0 to disable the fallback (GPU scatter), generally not recommended.
CUDA_NO_SCATTER="${PPML_CUDA_NO_SCATTER:-1}"

# 【2026-09-20 新增】staging async（默认开 ✓；=0 回落到原阻塞拷贝路径）
STAGING_ASYNC="${PPML_STAGING_ASYNC:-1}"
# 显存预算（MB）；空 = 后端自定（空闲显存×4/5）
GPU_BUDGET_MB="${PPML_GPU_BUDGET_MB:-}"
# 【2026-09-20 新增】打印：splits + alloc（RFAA_LEAN=1 可一键全关 ✓）
if [ "${RFAA_LEAN:-0}" = "1" ]; then
    DBG_SPLITS=0; DBG_GALLOCR=0; DBG_ALLOC=0; DBG_SCHED_VERBOSE=0; DBG_STAGING=0; DBG_CTX=0
else
    DBG_SPLITS="${GRAPH_DEBUG_SCHED:-1}"                  # split 级：split_graph / PLAN / COMPUTE / backend[] / budget
    DBG_GALLOCR="${GRAPH_DEBUG_GALLOCR:-1}"               # [gallocr] reserve done
    DBG_ALLOC="${PPML_DEBUG_ALLOC:-1}"                    # [alloc-buffer] 每次分配
    DBG_SCHED_VERBOSE="${GRAPH_DEBUG_SCHED_VERBOSE:-0}"   # 逐节点 dump（很吵 ⚠️）
    DBG_STAGING="${GRAPH_DEBUG_STAGING:-0}"               # 逐条 staging 拷贝（很吵 ⚠️）
    # 【2026-09-21】每样本上下文对象数（内存自检 ✓）：开销≈每样本一行，建议长训练留着 ✓
    DBG_CTX="${PPML_DEBUG_CTX:-1}"
fi

# ---------- Log ----------
LOG="remote_fulltrain_$(date +%Y%m%d_%H%M%S).log"

# ---------- Pre-flight checks ----------
if [ ! -x "$BIN" ]; then
    echo "[FATAL] executable not found: $BIN" | tee -a "$LOG"
    echo "        build it on the server first: cd $PROJ && bash build_remote.sh ppml_train" | tee -a "$LOG"
    exit 1
fi
if [ ! -f "$PROJ/data/P04637_alignment.a3m" ]; then
    echo "[FATAL] missing P04637 data (data/P04637_alignment.a3m)" | tee -a "$LOG"
    exit 1
fi
if [ ! -d "$PROJ/data/training_batch_data" ]; then
    echo "[FATAL] missing multi-sample dataset (data/training_batch_data)" | tee -a "$LOG"
    exit 1
fi
# Stage 2 default subset (L<=103): generated by python_scripts/filter_samples_by_len.py
if [ ! -d "$PROJ/data/training_batch_data_le103" ]; then
    echo "[FATAL] missing multi-sample subset dataset (data/training_batch_data_le103, L<=103)" | tee -a "$LOG"
    echo "        run first: python3 python_scripts/filter_samples_by_len.py 103" | tee -a "$LOG"
    exit 1
fi

# ---------- Env export ----------
export FULL_TRAIN PPML_NUM_EPOCHS="$NUM_EPOCHS" PPML_LR="$LR" PPML_CLIP_NORM="$CLIP" \
       PPML_ACCUM="$ACCUM" PPML_MSA_DEPTH="$MSA_DEPTH" \
       PPML_USE_CUDA="$USE_CUDA" PPML_CUDA_SCHED="$CUDA_SCHED" \
       PPML_CUDA_NO_SCATTER="$CUDA_NO_SCATTER" \
       PPML_STAGING_ASYNC="$STAGING_ASYNC" \
       GRAPH_DEBUG_SCHED="$DBG_SPLITS" GRAPH_DEBUG_GALLOCR="$DBG_GALLOCR" \
       PPML_DEBUG_ALLOC="$DBG_ALLOC" \
       PPML_DEBUG_CTX="$DBG_CTX"
# 可选：显存预算（非空才导出 ✓；不导出则后端用 空闲显存×4/5）
if [ -n "$GPU_BUDGET_MB" ]; then
    export PPML_GPU_BUDGET_MB="$GPU_BUDGET_MB"
fi
# 可选：逐节点 dump / 逐条 staging（默认 0 = 关 ✓，避免长训练日志爆量 ✗）
if [ "$DBG_SCHED_VERBOSE" != "0" ]; then export GRAPH_DEBUG_SCHED_VERBOSE=1; fi
if [ "$DBG_STAGING" != "0" ]; then export GRAPH_DEBUG_STAGING=1; fi
echo "[env] staging_async=$STAGING_ASYNC | budget=${GPU_BUDGET_MB:-auto(4/5 free)} | prints: splits=$DBG_SPLITS gallocr=$DBG_GALLOCR alloc=$DBG_ALLOC verbose=$DBG_SCHED_VERBOSE staging_dbg=$DBG_STAGING ctx=$DBG_CTX" | tee -a "$LOG"
export LD_LIBRARY_PATH="$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -n "$STDCPP_PRELOAD" ] && [ -f "$STDCPP_PRELOAD" ]; then
    export LD_PRELOAD="$STDCPP_PRELOAD"
    echo "[env] LD_PRELOAD=$LD_PRELOAD" | tee -a "$LOG"
else
    echo "[env] LD_PRELOAD not set (STDCPP_PRELOAD empty or file missing)" | tee -a "$LOG"
fi
echo "[env] LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "FULL_TRAIN remote training   log: $LOG" | tee -a "$LOG"
echo "hyper-params: epochs=$NUM_EPOCHS lr=$LR clip=$CLIP accum=$ACCUM msa=stage1-256/stage2-$MULTI_MSA_DEPTH" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# ---------- Stage 1: single sample P04637 (with templates) ----------
echo "" | tee -a "$LOG"
echo "########## Stage 1/2: P04637 single-sample training (with templates) ##########" | tee -a "$LOG"
# Note: multi-sample mode does not support templates (load_from_files template arg=""),
#       so P04637 uses the single-sample path: template = P04637_template_coords,
#       true coords = P04637_pdbs (multiple PDBs auto-merged).
#       2026-08-28 onward skipped by default (high memory peak; see diag_mem: pair track L^2 dominates, MSA-depth independent);
#       set SKIP_P04637=0 to run stage 1.
if [ "${SKIP_P04637:-1}" = "1" ]; then
    echo "[skip] SKIP_P04637=1, skipping stage 1" | tee -a "$LOG"
else
    cd "$PROJ"
    # disable multi-sample, use single-sample path with templates
    unset PPML_MULTI_SAMPLE
    # stage 1 MSA depth = 256 (overrides global MSA_DEPTH; stage 2 uses MULTI_MSA_DEPTH, default also 256)
    export PPML_MSA_DEPTH=256
    # stage 1 has no timeout: FULL_TRAIN full-size first build + N=256 single epoch can far exceed 1 hour,
    # a hard timeout (exit 124) would kill training and lose all progress (no checkpoint saved). Let it finish naturally.
    echo "[stage1] no timeout, ends when training finishes" | tee -a "$LOG"
    echo "[stage1] progress: tail -f $LOG and watch 'Epoch N/M completed ... loss:' lines" | tee -a "$LOG"
    echo "[stage1] if no Epoch output for a long time, check whether '[FWD-GRAPH] ... build ...' was printed (slow first build is normal)" | tee -a "$LOG"
    "$BIN" \
        "$PROJ/data/P04637_alignment.a3m" \
        "$PROJ/data/P04637.fasta" \
        "$PROJ/data/P04637_pdbs" \
        "$PROJ/data/P04637_template_coords" \
        "" 2>&1 | tee -a "$LOG"
    st1=${PIPESTATUS[0]}
    echo "[stage1] exit=$st1" | tee -a "$LOG"
    if [ "$st1" -ne 0 ] && [ "$st1" -ne 2 ]; then
        echo "[WARN] stage 1 failed (exit=$st1), continuing to stage 2" | tee -a "$LOG"
    fi
fi

# ---------- Stage 2: multi-sample (only L<=103 samples) ----------
echo "" | tee -a "$LOG"
echo "########## Stage 2/2: multi-sample training (training_batch_data_le103, L<=103) ##########" | tee -a "$LOG"
if [ "${SKIP_MULTI:-0}" = "1" ]; then
    echo "[skip] SKIP_MULTI=1, skipping stage 2" | tee -a "$LOG"
else
    cd "$PROJ"
    # stage 2 MSA depth = MULTI_MSA_DEPTH (2026-09-11 lowered global 512 -> 256; see header/var comments)
    export PPML_MSA_DEPTH="$MULTI_MSA_DEPTH"
    echo "[multi] MSA depth = $MULTI_MSA_DEPTH (override via PPML_MULTI_MSA_DEPTH)" | tee -a "$LOG"
    export PPML_MULTI_SAMPLE=1
    # 2026-08-28: use only samples with length <= 103 (pair track L^2 dominates memory; long samples peak high).
    #   The subset dir is generated by python_scripts/filter_samples_by_len.py, contains P62805(L=103)+P62891(L=51).
    #   To switch back to the full set: PPML_DATASET_DIR=$PROJ/data/training_batch_data
    export PPML_DATASET_DIR="${PPML_DATASET_DIR:-$PROJ/data/training_batch_data_le103}"
    echo "[multi] dataset: $PPML_DATASET_DIR" | tee -a "$LOG"
    # 【2026-09-20】stage 2 的 CUDA / staging / 打印开关回显（便于事后核对这次跑的是什么）
    echo "[multi] CUDA=$USE_CUDA sched=$CUDA_SCHED staging_async=$STAGING_ASYNC budget=${GPU_BUDGET_MB:-auto}MB" | tee -a "$LOG"
    echo "[multi] prints: splits=$DBG_SPLITS gallocr=$DBG_GALLOCR alloc=$DBG_ALLOC（verbose=$DBG_SCHED_VERBOSE staging_dbg=$DBG_STAGING）" | tee -a "$LOG"
    echo "[multi] 行样例：'[sched] COMPUTE split=7 backend=0(CUDA) i=[..) nodes=210 ops=[…]'、'[sched] split_graph: pre-merge … after merge …'、'[alloc-buffer] CUDA usage=COMPUTE size=1.38 GB'、'[gallocr] reserve done: backend=0 peak=… GB'" | tee -a "$LOG"
    echo "[multi] ⚠️ 全量打印会让日志变大：需要精简用 RFAA_LEAN=1（或 PPML_DEBUG_ALLOC=0 只保留 ≥1GB）" | tee -a "$LOG"
    # 【2026-09-21】内存自检提示（方案 B：grad 张量化 ⇒ 每样本释放 ✓）
    echo "[multi] 内存自检：启动后应看到 '[grad-carrier] 持久梯度载体图已建：params=N/N' ✓（=每样本即可释放样本图 ✓）" | tee -a "$LOG"
    echo "[multi] 内存自检：PPML_DEBUG_CTX=$DBG_CTX ⇒ 每样本一行 '[ctx] … n_objects=…'，**应持平** ✓；单调增长 ✗ = 仍在累积（跑完脚本会自动判定 ✓）" | tee -a "$LOG"
    echo "[multi] ⚠️ 若这两类行都没有 ⇒ 服务器二进制是旧版：git pull && bash build_remote.sh ppml_train（旧版会在 accum 窗口内累积 ⇒ Context memory exhausted / exit 134 ✗）" | tee -a "$LOG"
    echo "[multi] no timeout, ends when training finishes" | tee -a "$LOG"
    # stage 2 has no timeout: 10 epochs of multi-sample training take a long time, ends naturally.
    "$BIN" \
        "" "" "" "" "" 2>&1 | tee -a "$LOG"
    st2=${PIPESTATUS[0]}
    echo "[stage2] exit=$st2" | tee -a "$LOG"

    # 【2026-09-20】stage 2 运行摘要：splits / 显存 / staging / epoch（全在 $LOG 里，这里把关键行汇总到日志尾 ✓）
    echo "[stage2] ---- 运行摘要（splits / 显存 / staging / epoch）----" | tee -a "$LOG"
    grep -aE 'split_graph: (pre-merge|after merge)|split_graph: n_backends' "$LOG" | tail -3 | tee -a "$LOG"
    echo "[stage2] split 数按后端（backend id：0=CUDA 最高优先级，1=CPU）：" | tee -a "$LOG"
    grep -a 'COMPUTE split=' "$LOG" | sed -E 's/.*backend=([0-9]+).*/\1/' | sort | uniq -c | tee -a "$LOG"
    grep -a 'budget 拒绝总数' "$LOG" | tail -1 | tee -a "$LOG"
    grep -a 'reserve done' "$LOG" | tail -4 | tee -a "$LOG"
    grep -a 'STAGING' "$LOG" | head -2 | tee -a "$LOG"
    grep -aE 'Epoch [0-9]+/[0-9]+ completed' "$LOG" | tail -5 | tee -a "$LOG"

    # ---- 【2026-09-21】内存自检（方案 B：grad 张量化 ⇒ 每样本释放 ✓）----
    #   ① 有没有 '[grad-carrier]'：没有 ⇒ 旧二进制（累积风险 ✗）；
    #   ② '[ctx] n_objects' 是否持平：峰值 > 首值×2+200 ⇒ 判定为仍在累积 ✗（给兜底开关建议 ✓）。
    #   注：先算完再打印（打印内容里含这些关键字，避免自我匹配 ✗）。
    echo "[stage2] ---- 内存自检（每样本释放 / 上下文对象数）----" | tee -a "$LOG"
    carrier_line=$(grep -a 'grad-carrier' "$LOG" | tail -1)
    ctx_n=$(grep -ac '\[ctx\]' "$LOG")
    ctx_first=$(grep -a '\[ctx\]' "$LOG" | head -1 | sed -E 's/.*n_objects=([0-9]+).*/\1/')
    ctx_max=$(grep -ao 'n_objects=[0-9]*' "$LOG" | sed 's/n_objects=//' | sort -n | tail -1)
    ctx_grow=$(awk -v a="${ctx_first:-0}" -v b="${ctx_max:-0}" 'BEGIN{print (b > a*2+200) ? 1 : 0}')
    if [ -n "$carrier_line" ]; then
        echo "[stage2] ✅ $carrier_line" | tee -a "$LOG"
    else
        echo "[stage2] ⚠️ 未找到 '[grad-carrier]' ⇒ 服务器二进制可能是旧版：git pull && bash build_remote.sh ppml_train 后重跑（旧版 accum 窗口内不释放 ⇒ 可能 exit 134 ✗）" | tee -a "$LOG"
    fi
    if [ "$ctx_n" -gt 0 ]; then
        echo "[stage2] [ctx] 行数=$ctx_n  首个样本 n_objects=${ctx_first:-?}  峰值=${ctx_max:-?}" | tee -a "$LOG"
        if [ "$ctx_grow" = "1" ]; then
            echo "[stage2] ⚠️ 对象数仍在累积（峰值 > 首值×2+200）⇒ 兜底：PPML_ACCUM=1（或 2）+ PPML_MULTI_MSA_DEPTH=128 重跑 ✓" | tee -a "$LOG"
        else
            echo "[stage2] ✅ 对象数基本持平 ⇒ 内存与 PPML_ACCUM 已解耦（长训练不会因 accum 窗口累积而爆 ✗→✓）" | tee -a "$LOG"
        fi
    else
        echo "[stage2] （没看到 '[ctx]' 行：本脚本默认 PPML_DEBUG_CTX=1 ✓；若被 RFAA_LEAN=1 关掉属预期 ✓）" | tee -a "$LOG"
    fi
    # multi-sample returns 2 when space is insufficient (stops without lowering config)
    if [ "$st2" = "2" ]; then
        echo "[ERROR] multi-sample stopped due to insufficient space (exit 2). Server free memory must be >= PPML_MIN_FREE_GB (default 22)" | tee -a "$LOG"
        exit 2
    fi
fi

echo "" | tee -a "$LOG"
echo "========== All stages finished ==========" | tee -a "$LOG"
echo "log: $LOG" | tee -a "$LOG"
exit 0
