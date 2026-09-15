#!/usr/bin/env bash
# RFAA-Cpp remote server build script (based on build_wsl.sh, auto-detects paths)
# Usage: bash build_remote.sh [target]
#   default target = ppml_train
# Env overrides:
#   RFAA_PROJECT   - project root (default: directory of this script)
#   RFAA_PY        - python interpreter (default: auto-detect conda base or python3)
#   RFAA_JOBS      - parallel jobs (default: nproc)
#   RFAA_CUDA_ARCH - CUDA architecture (default: auto-detect - clear cache then CMake detects via nvidia-smi;
#                    explicit value overrides auto, e.g. RFAA_CUDA_ARCH=89)

set -euo pipefail

# ---------- Path config ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${RFAA_PROJECT:-$SCRIPT_DIR}"
BUILD_DIR="$PROJECT_DIR/build/remote"
JOBS="${RFAA_JOBS:-$(nproc 2>/dev/null || echo 8)}"

# ---------- Python location (auto-detect) ----------
PY="${RFAA_PY:-}"
if [ -z "$PY" ]; then
    # prefer conda base, then system python3
    for cand in "$CONDA_PREFIX/bin/python3" "$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3" \
                "/opt/conda/bin/python3" "/root/anaconda3/bin/python3" "/usr/bin/python3"; do
        if [ -x "$cand" ]; then PY="$cand"; break; fi
    done
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "error: python not found. Set RFAA_PY or install python3" >&2
    exit 1
fi

# Build target, default ppml_train
TARGET="${1:-ppml_train}"

cd "$PROJECT_DIR"

echo "==== 1. Resolve python path ===="
echo "  PROJECT = $PROJECT_DIR"
echo "  PY      = $PY"
echo "  JOBS    = $JOBS"

# Auto-derive include/lib path and name from python
PY_INC=$("$PY" -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_LIBDIR=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PY_LIBNAME=$("$PY" -c "import sysconfig; print('libpython'+sysconfig.get_config_var('VERSION')+'.so')")
PY_LIB="$PY_LIBDIR/$PY_LIBNAME"

echo "  INCLUDE  = $PY_INC"
echo "  LIB      = $PY_LIB"

# Verify the include dir exists; fall back if the sysconfig value is missing
if [ ! -d "$PY_INC" ]; then
    echo "  [!] include dir $PY_INC not found, trying fallback"
    PY_INC="$PY_LIBDIR/../include/$(basename "$PY_INC")"
    echo "      using $PY_INC"
fi

echo ""
echo "==== 2. Create build dir ===="
mkdir -p "$BUILD_DIR"

echo ""
echo "==== 3. CUDA architecture auto-detect ===="
# Priority: explicit RFAA_CUDA_ARCH > nvidia-smi detect > CMake fallback set (80;86;89;90)
# Default clears the cache (-U) so CMake re-detects each time (avoids stale arch from cache).
if [ -n "${RFAA_CUDA_ARCH:-}" ]; then
    echo "  [cuda] explicit arch: $RFAA_CUDA_ARCH (overrides auto-detect)"
    CMAKE_ARCH_ARG=(-DCMAKE_CUDA_ARCHITECTURES="$RFAA_CUDA_ARCH")
else
    echo "  [cuda] clearing cache, CMake auto-detect (nvidia-smi -> fallback 80;86;89;90)"
    CMAKE_ARCH_ARG=(-U CMAKE_CUDA_ARCHITECTURES)
    if command -v nvidia-smi >/dev/null 2>&1; then
        CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        echo "  [cuda] nvidia-smi compute_cap = ${CAP:-detect failed}"
    else
        echo "  [cuda] nvidia-smi not found (no NVIDIA GPU / driver), will use fallback arch set"
    fi
fi

echo ""
echo "==== 4. cmake configure ===="
CMAKE_ARGS=(
    -S "$PROJECT_DIR" -B "$BUILD_DIR"
    -DPython3_EXECUTABLE="$PY"
    -DPython3_INCLUDE_DIR="$PY_INC"
    -DPython3_LIBRARY="$PY_LIB"
    "${CMAKE_ARCH_ARG[@]}"
)
cmake "${CMAKE_ARGS[@]}"
echo "  configure done"

echo ""
echo "==== 5. Build target: $TARGET ===="
cmake --build "$BUILD_DIR" --target "$TARGET" -j"$JOBS"

echo ""
echo "==== Done ===="
echo "executable: $BUILD_DIR/examples/$TARGET"
echo "run example (with python lib and libstdc++ preload):"
echo "  LD_PRELOAD=\$(gcc -print-file-name=libstdc++.so.6) \\"
echo "    LD_LIBRARY_PATH=$PY_LIBDIR $BUILD_DIR/examples/$TARGET data/training_batch_data/P62891_alignment.a3m data/P62891.fasta data/training_batch_data/4ug0_P62891_mapping.csv \"\""
echo "  if 'GLIBCXX_3.4.30 not found' -> preload system libstdc++ (see RFAA_STDCPP in remote_fulltrain.sh)"
