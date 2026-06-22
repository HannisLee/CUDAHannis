import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

QWEN35_9B_BATCH = 1
QWEN35_9B_SEQUENCE_LENGTH = 1024
QWEN35_9B_INTERMEDIATE_SIZE = 12288

from kernels.activation_24 import (
    activation_24_sparsity_cuda,
    activation_24_sparsity_pytorch,
    activation_24_sparsity_triton,
)


def time_cuda(fn, warmup: int = 3, repeat: int = 10) -> float:
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


def check_24_sparsity(out: torch.Tensor) -> bool:
    last_dim = out.shape[-1]
    padded_last_dim = ((last_dim + 3) // 4) * 4
    pad = padded_last_dim - last_dim
    padded = torch.nn.functional.pad(out, (0, pad)) if pad else out
    grouped = padded.reshape(-1, padded_last_dim // 4, 4)
    return bool(((grouped != 0).sum(dim=-1) <= 2).all().item())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    shape = (QWEN35_9B_BATCH, QWEN35_9B_SEQUENCE_LENGTH, QWEN35_9B_INTERMEDIATE_SIZE)
    print("operator: activation 2:4 sparsity")
    print("model shape: Qwen3.5-9B MLP intermediate activation")
    print(f"shape: {shape}")

    for dtype in (torch.float32, torch.float16):
        torch.manual_seed(24)
        x = torch.randn(*shape, device="cuda", dtype=dtype)

        ref = activation_24_sparsity_pytorch(x)
        triton_out = activation_24_sparsity_triton(x)
        cuda_out = activation_24_sparsity_cuda(x)

        assert torch.equal(triton_out, ref)
        assert torch.equal(cuda_out, ref)
        assert check_24_sparsity(ref)

        pytorch_ms = time_cuda(lambda: activation_24_sparsity_pytorch(x))
        triton_ms = time_cuda(lambda: activation_24_sparsity_triton(x))
        cuda_ms = time_cuda(lambda: activation_24_sparsity_cuda(x))

        print(f"\ndtype: {dtype}")
        print(f"triton max error: {(triton_out - ref).abs().max().item():.8e}")
        print(f"cuda max error:   {(cuda_out - ref).abs().max().item():.8e}")
        print(f"2:4 valid:        {check_24_sparsity(ref)}")
        print(f"PyTorch:          {pytorch_ms:.4f} ms")
        print(f"Triton:           {triton_ms:.4f} ms")
        print(f"CUDA extension:   {cuda_ms:.4f} ms")

    print("\nPASS")


if __name__ == "__main__":
    main()
