"""Correctness + micro-benchmark for the RMSNorm backends.

Runs the PyTorch reference, the Triton kernel and every applicable CUDA variant
from rms_norm_f16_f32.cu for float32 and float16, at a realistic Qwen3-shaped
hidden size (K=4096) and at K=1024 where all variants are available.

Run:  /home/han.li/miniconda3/envs/triton-cu118/bin/python kernels/rms_norm/compare_single.py
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# RMSNorm operates on the hidden size. Qwen3-8B hidden_size = 4096.
N_ROWS = 4096
CONFIGS = [
    (N_ROWS, 1024),  # all CUDA variants available
    (N_ROWS, 4096),  # Qwen3-8B hidden size (subset of variants)
]
SCALE_G = 1.0

from kernels.rms_norm import rms_norm_cuda, rms_norm_pytorch, rms_norm_triton
from kernels.rms_norm.rms_norm_common import REDUCE_DTYPE, variants_for


def time_cuda(fn, warmup: int = 5, repeat: int = 20) -> float:
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


def tolerances(dtype: torch.dtype, reduce: str) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-3, 1e-3
    # fp16 output is quantized to ~1e-3 relative; fp16 reductions add more error.
    if reduce == "f32":
        return 1e-2, 1e-2
    return 5e-2, 5e-2


def bytes_touched(n: int, k: int, dtype: torch.dtype) -> int:
    # read x once + write y once
    return 2 * n * k * dtype.itemsize


def run_config(n: int, k: int) -> None:
    print(f"\nshape: ({n}, {k})")
    for dtype in (torch.float32, torch.float16):
        torch.manual_seed(4)
        x = (torch.randn(n, k, device="cuda", dtype=dtype) * 3.0).contiguous()
        ref = rms_norm_pytorch(x, SCALE_G)

        # Triton (fp32 accumulation, dtype-honouring)
        tri = rms_norm_triton(x, SCALE_G)
        tri_max = (tri - ref).abs().max().item()
        assert torch.allclose(tri, ref, rtol=1e-2, atol=1e-2), (
            f"triton mismatch (dtype={dtype}): max err {tri_max:.3e}"
        )

        print(f"\n  dtype: {dtype}")
        print(f"  {'backend':<22}{'max err':>14}{'time (ms)':>12}{'BW (GB/s)':>12}")
        tri_ms = time_cuda(lambda: rms_norm_triton(x, SCALE_G))
        bw = bytes_touched(n, k, dtype) / (tri_ms * 1e-3) / 1e9
        print(f"  {'pytorch(ref)':<22}{'-':>14}{'-':>12}{'-':>12}")
        print(f"  {'triton':<22}{tri_max:>14.3e}{tri_ms:>12.4f}{bw:>12.1f}")

        for name in variants_for(dtype, k):
            reduce = REDUCE_DTYPE[name]
            out = rms_norm_cuda(x, SCALE_G, variant=name)
            max_err = (out - ref).abs().max().item()
            rtol, atol = tolerances(dtype, reduce)
            assert torch.allclose(out, ref, rtol=rtol, atol=atol), (
                f"cuda {name} mismatch (dtype={dtype}, reduce={reduce}): "
                f"max err {max_err:.3e} > atol {atol:.0e}"
            )
            ms = time_cuda(lambda: rms_norm_cuda(x, SCALE_G, variant=name))
            bw = bytes_touched(n, k, dtype) / (ms * 1e-3) / 1e9
            label = f"cuda {name}"
            if reduce == "f16":
                label += " (f16)"
            print(f"  {label:<22}{max_err:>14.3e}{ms:>12.4f}{bw:>12.1f}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("operator: RMSNorm  (y = x * rsqrt(mean(x^2) + eps) * g)")
    print(f"scale g: {SCALE_G}, eps: 1e-5")
    for n, k in CONFIGS:
        run_config(n, k)

    print("\nPASS")


if __name__ == "__main__":
    main()
