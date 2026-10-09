#!/usr/bin/env bash
# 必须 source；所有运行缓存显式落在本项目，避免污染用户环境和旧项目。
GPACO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export GPACO_ROOT
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$GPACO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$GPACO_ROOT/.cache"
export TMPDIR="$GPACO_ROOT/.cache/tmp"
export PIP_CACHE_DIR="$GPACO_ROOT/.cache/pip"
export MAMBA_ROOT_PREFIX="$GPACO_ROOT/.tools/mamba"
export CUDA_CACHE_PATH="$GPACO_ROOT/.cache/cuda-driver/$(hostname -s)"
export CUPY_CACHE_DIR="$GPACO_ROOT/.cache/cupy/$(hostname -s)"
export NUMBA_CACHE_DIR="$GPACO_ROOT/.cache/numba/$(hostname -s)"
export TORCH_EXTENSIONS_DIR="$GPACO_ROOT/.cache/torch-extensions/$(hostname -s)"
export TRITON_CACHE_DIR="$GPACO_ROOT/.cache/triton/$(hostname -s)"
export MPLCONFIGDIR="$GPACO_ROOT/.cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$GPACO_ROOT/.envs/main/bin:$GPACO_ROOT/.tools/bin:$PATH"
export CUDA_HOME="$GPACO_ROOT/.envs/main"
export CUDA_PATH="$CUDA_HOME"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-c++"
export NVCC_CCBIN="$CXX"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
