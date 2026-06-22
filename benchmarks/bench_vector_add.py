import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.vector_add import triton_add

try:
    from kernels.vector_add import cuda_add
except Exception:
    cuda_add = None


def time_cuda(fn, warmup: int = 20, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请先运行 source scripts/activate_env.sh 并检查 GPU/driver/PyTorch CUDA。")

    n = 16 * 1024 * 1024
    dtype = torch.float32
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)

    cuda_ms = None
    cuda_error = None
    if cuda_add is not None:
        try:
            cuda_add(x, y)
            torch.cuda.synchronize()
            cuda_ms = time_cuda(lambda: cuda_add(x, y))
        except Exception as exc:
            cuda_error = exc

    torch_ms = time_cuda(lambda: x + y)
    triton_ms = time_cuda(lambda: triton_add(x, y))

    bytes_moved = 3 * x.numel() * x.element_size()
    torch_bw = bytes_moved / (torch_ms / 1_000) / 1e9
    triton_bw = bytes_moved / (triton_ms / 1_000) / 1e9
    cuda_bw = bytes_moved / (cuda_ms / 1_000) / 1e9 if cuda_ms is not None else None

    print(f"elements: {n}")
    print(f"dtype: {dtype}")
    print(f"PyTorch x + y: {torch_ms:.4f} ms, bandwidth: {torch_bw:.2f} GB/s")
    print(f"Triton add:    {triton_ms:.4f} ms, bandwidth: {triton_bw:.2f} GB/s")
    if cuda_ms is not None and cuda_bw is not None:
        print(f"CUDA add:      {cuda_ms:.4f} ms, bandwidth: {cuda_bw:.2f} GB/s")
    elif cuda_error is not None:
        print(f"CUDA add:      skipped ({cuda_error})")


if __name__ == "__main__":
    main()
