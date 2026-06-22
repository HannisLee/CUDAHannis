import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

QWEN35_9B_BATCH = 1
QWEN35_9B_SEQUENCE_LENGTH = 1024
QWEN35_9B_INTERMEDIATE_SIZE = 12288

from kernels.vector_add import cuda_add, triton_add


def time_cuda(fn, warmup: int = 3, repeat: int = 20) -> float:
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
        raise RuntimeError("CUDA is not available.")

    n = QWEN35_9B_BATCH * QWEN35_9B_SEQUENCE_LENGTH * QWEN35_9B_INTERMEDIATE_SIZE
    print("operator: vector add")
    print("model shape: Qwen3.5-9B MLP intermediate activation")
    print(f"elements: {n}")

    for dtype in (torch.float32, torch.float16):
        torch.manual_seed(1)
        x = torch.randn(n, device="cuda", dtype=dtype)
        y = torch.randn(n, device="cuda", dtype=dtype)

        ref = x + y
        triton_out = triton_add(x, y)
        cuda_out = cuda_add(x, y)

        rtol = 1e-6 if dtype == torch.float32 else 1e-3
        atol = 1e-6 if dtype == torch.float32 else 1e-3
        assert torch.allclose(triton_out, ref, rtol=rtol, atol=atol)
        assert torch.allclose(cuda_out, ref, rtol=rtol, atol=atol)

        pytorch_ms = time_cuda(lambda: x + y)
        triton_ms = time_cuda(lambda: triton_add(x, y))
        cuda_ms = time_cuda(lambda: cuda_add(x, y))

        print(f"\ndtype: {dtype}")
        print(f"triton max error: {(triton_out - ref).abs().max().item():.8e}")
        print(f"cuda max error:   {(cuda_out - ref).abs().max().item():.8e}")
        print(f"PyTorch:          {pytorch_ms:.4f} ms")
        print(f"Triton:           {triton_ms:.4f} ms")
        print(f"CUDA extension:   {cuda_ms:.4f} ms")

    print("\nPASS")


if __name__ == "__main__":
    main()
