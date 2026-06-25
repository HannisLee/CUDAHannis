#!/usr/bin/env bash

ENV_NAME="vllm-cu129"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda not found. Please install Miniconda/Anaconda or load conda first." >&2
    return 1 2>/dev/null || exit 1
  fi
else
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi
fi

conda activate "$ENV_NAME" || return 1 2>/dev/null || exit 1

# nvcc is pip-installed (nvidia-cuda-nvcc) under site-packages/nvidia/cu13,
# not at $CONDA_PREFIX/bin/nvcc. Locate it and point CUDA_HOME at it so
# torch.utils.cpp_extension picks up the nvcc matching torch's CUDA build.
if [ -n "${CONDA_PREFIX:-}" ]; then
  PIP_CUDA_HOME="$("$CONDA_PREFIX/bin/python" - <<'PY'
import os
try:
    import nvidia
except ImportError:
    raise SystemExit
for root in nvidia.__path__:
    for sub in sorted(os.listdir(root)):
        cand = os.path.join(root, sub)
        if os.path.exists(os.path.join(cand, "bin", "nvcc")):
            print(cand)
            raise SystemExit
PY
)"
  if [ -n "$PIP_CUDA_HOME" ]; then
    export CUDA_HOME="$PIP_CUDA_HOME"
  fi
fi

# vllm-cu129 ships no conda gcc/g++; use the system compiler as host compiler.
if command -v gcc >/dev/null 2>&1; then
  export CC="$(command -v gcc)"
fi
if command -v g++ >/dev/null 2>&1; then
  export CXX="$(command -v g++)"
fi

HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
  CACHE_ROOT="/scratch/$USER/cudahannis/$HOST_SHORT"
else
  CACHE_ROOT="/tmp/$USER/cudahannis/$HOST_SHORT"
fi

export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"

python - <<'PY'
import os
import socket

print("hostname:", socket.gethostname())
print("conda env:", os.environ.get("CONDA_DEFAULT_ENV", "unknown"))
print("TRITON_CACHE_DIR:", os.environ.get("TRITON_CACHE_DIR", ""))
print("TORCH_EXTENSIONS_DIR:", os.environ.get("TORCH_EXTENSIONS_DIR", ""))
print("CUDA_CACHE_PATH:", os.environ.get("CUDA_CACHE_PATH", ""))
print("CUDA_HOME:", os.environ.get("CUDA_HOME", ""))
print("CC:", os.environ.get("CC", ""))
print("CXX:", os.environ.get("CXX", ""))

try:
    import torch
    print("torch version:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("GPU name: CUDA not available")
except Exception as exc:
    print("torch import failed:", repr(exc))

try:
    import triton
    print("triton version:", triton.__version__)
except Exception as exc:
    print("triton import failed:", repr(exc))
PY
