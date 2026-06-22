#!/usr/bin/env bash
set -u

section() {
  printf '\n=== %s ===\n' "$1"
}

section "基本信息"
printf 'hostname: %s\n' "$(hostname)"
printf 'pwd: %s\n' "$(pwd)"
printf 'shell: %s\n' "${SHELL:-unknown}"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  printf 'os: %s\n' "${PRETTY_NAME:-unknown}"
else
  uname -a
fi

section "conda"
if command -v conda >/dev/null 2>&1; then
  command -v conda
  conda info --envs
else
  printf 'conda: not found\n'
fi

section "nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  command -v nvidia-smi
  nvidia-smi
  printf '\n--- GPU summary ---\n'
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  printf 'nvidia-smi: not found\n'
fi

section "nvcc"
if command -v nvcc >/dev/null 2>&1; then
  command -v nvcc
  nvcc --version
else
  printf 'nvcc: not found\n'
fi

section "CUDA 相关环境变量"
printf 'CUDA_HOME=%s\n' "${CUDA_HOME:-}"
printf 'CUDA_PATH=%s\n' "${CUDA_PATH:-}"
printf 'PATH entries containing cuda/nsight/nvidia:\n'
printf '%s\n' "${PATH:-}" | tr ':' '\n' | grep -Ei 'cuda|nsight|nvidia' || true
printf 'LD_LIBRARY_PATH entries containing cuda/nsight/nvidia:\n'
printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -Ei 'cuda|nsight|nvidia' || true

section "/usr/local cuda 目录"
find /usr/local -maxdepth 1 -type d -name 'cuda*' -printf '%f -> %p\n' 2>/dev/null | sort || true

section "Nsight 工具"
if command -v nsys >/dev/null 2>&1; then
  command -v nsys
  nsys --version || true
else
  printf 'nsys: not found\n'
fi
if command -v ncu >/dev/null 2>&1; then
  command -v ncu
  ncu --version || true
else
  printf 'ncu: not found\n'
fi

section "PyTorch / Triton"
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("torch cuda available:", torch.cuda.is_available())
    print("torch device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("torch device 0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch import failed:", repr(exc))

try:
    import triton
    print("triton:", triton.__version__)
except Exception as exc:
    print("triton import failed:", repr(exc))
PY

section "说明"
printf 'nvidia-smi 里的 CUDA Version 表示当前 driver 支持的最高 CUDA runtime 版本，不一定等于本机安装的 CUDA toolkit/nvcc 版本。\n'
