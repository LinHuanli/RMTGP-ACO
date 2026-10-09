#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$GPACO_ROOT"
if [[ -x .envs/main/bin/python ]]; then
    echo "项目环境已存在；不覆盖可能正在运行的环境。"
    python -m pip check
    exit 0
fi
GPACO_SETUP_REPORT="$GPACO_ROOT/artifacts/provenance/environments/main/p01"
if [[ -e "$GPACO_SETUP_REPORT" ]]; then
    echo "环境记录已存在；先登记新环境版本，不覆盖旧清单。" >&2
    exit 1
fi
mkdir -p .cache/downloads .tools "$GPACO_SETUP_REPORT"
if [[ ! -x .tools/bin/micromamba ]]; then
    curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o .cache/downloads/micromamba.tar.bz2
    tar -xjf .cache/downloads/micromamba.tar.bz2 -C .tools bin/micromamba
fi
# NVCC、GCC 与 Python 同一前缀；不沿用项目外的 CUDA 12.6。
micromamba create -y -p "$GPACO_ROOT/.envs/main" -c conda-forge -c nvidia \
    'python=3.12' pip 'gcc_linux-64=14' 'gxx_linux-64=14' 'cuda-toolkit=13.2' ninja cmake
export CUDA_HOME="$GPACO_ROOT/.envs/main"
export CUDA_PATH="$CUDA_HOME"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-c++"
export NVCC_CCBIN="$CXX"
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e '.[cuda,dev]'
python -m pip check
python -m pip freeze > "$GPACO_SETUP_REPORT/pip-freeze.txt"
micromamba list -p "$GPACO_ROOT/.envs/main" --explicit > "$GPACO_SETUP_REPORT/conda-explicit.txt"
python -c 'import numpy,numba,llvmlite,torch,deap,cupy; print(numpy.__version__,numba.__version__,llvmlite.__version__,torch.__version__,deap.__version__,cupy.__version__)'
