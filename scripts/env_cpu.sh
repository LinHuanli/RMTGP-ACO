#!/usr/bin/env bash
# 仅配置项目路径与缓存；不要求 CUDA / Torch，也不加载任何集群模块。
GPACO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export GPACO_ROOT PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$GPACO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$GPACO_ROOT/.cache"
export TMPDIR="$GPACO_ROOT/.cache/tmp"
export PIP_CACHE_DIR="$GPACO_ROOT/.cache/pip"
export NUMBA_CACHE_DIR="$GPACO_ROOT/.cache/numba-cpu/$(hostname -s)"
export MPLCONFIGDIR="$GPACO_ROOT/.cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
