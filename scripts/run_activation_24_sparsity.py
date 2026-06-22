import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.activation_24 import (
    activation_24_sparsity_cuda,
    activation_24_sparsity_pytorch,
    activation_24_sparsity_triton,
)


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


def check_24_sparsity(out: torch.Tensor) -> bool:
    last_dim = out.shape[-1]
    padded_last_dim = ((last_dim + 3) // 4) * 4
    pad = padded_last_dim - last_dim
    padded = torch.nn.functional.pad(out, (0, pad)) if pad else out
    grouped = padded.reshape(-1, padded_last_dim // 4, 4)
    counts = (grouped != 0).sum(dim=-1)
    return bool((counts <= 2).all().item())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请先运行 source scripts/activate_env.sh 并检查 GPU/driver/PyTorch CUDA。")

    shape = (4, 16, 4099)
    print(f"input shape: {shape}")

    for dtype in (torch.float32, torch.float16):
        torch.manual_seed(24)
        x = torch.randn(*shape, device="cuda", dtype=dtype)

        expected = activation_24_sparsity_pytorch(x)
        triton_out = activation_24_sparsity_triton(x)
        cuda_out = activation_24_sparsity_cuda(x)

        triton_error = (triton_out - expected).abs().max().item()
        cuda_error = (cuda_out - expected).abs().max().item()
        assert torch.equal(triton_out, expected)
        assert torch.equal(cuda_out, expected)
        assert check_24_sparsity(expected)

        pytorch_ms = time_cuda(lambda: activation_24_sparsity_pytorch(x), warmup=10, repeat=30)
        triton_ms = time_cuda(lambda: activation_24_sparsity_triton(x))
        cuda_ms = time_cuda(lambda: activation_24_sparsity_cuda(x))

        print(f"\ndtype: {dtype}")
        print(f"output shape: {tuple(expected.shape)}")
        print(f"2:4 sparsity valid: {check_24_sparsity(expected)}")
        print(f"triton max error: {triton_error:.8e}")
        print(f"cuda max error: {cuda_error:.8e}")
        print(f"PyTorch reference: {pytorch_ms:.4f} ms")
        print(f"Triton kernel:     {triton_ms:.4f} ms")
        print(f"CUDA extension:    {cuda_ms:.4f} ms")

    print("\nActivation 2:4 sparsity single script: PASS")


if __name__ == "__main__":
    main()
