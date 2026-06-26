import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from sigmoid_pytorch import sigmoid_pytorch
from sigmoid_triton import sigmoid_triton

lib = load(
    name="sigmoid_lib",
    sources=[str(CURRENT_DIR / "sigmoid.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: Callable,
    x: torch.Tensor,
    tag: str,
    ref: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    has_out = out is not None

    if has_out:
        out.fill_(0)

    # warmup
    if has_out:
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            out = perf_func(x)

    torch.cuda.synchronize()

    start = time.perf_counter()

    if has_out:
        for _ in range(iters):
            perf_func(x, out)
    else:
        for _ in range(iters):
            out = perf_func(x)

    torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = (end - start) * 1000
    mean_time = total_time / iters

    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(float(v), 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]

    if ref is None:
        max_abs = 0.0
    else:
        max_abs = (out - ref).abs().max().item()

    print(f"{out_info:>20}: {out_val}, time:{mean_time:.8f}ms, max_abs:{max_abs:.3e}")

    if show_all:
        print(out)

    return out, mean_time


def bench_shape(N: int, K: int):
    print("-" * 105)
    print(" " * 40 + f"N={N}, K={K}, dtype=fp16")
    print("-" * 105)

    x = torch.randn((N, K), device="cuda", dtype=torch.float16).contiguous()
    out = torch.zeros_like(x).contiguous()

    ref, _ = run_benchmark(sigmoid_pytorch, x, "torch")
    run_benchmark(lib.sigmoid_f16, x, "f16", ref, out)
    run_benchmark(lib.sigmoid_f16x8, x, "f16x8", ref, out)
    run_benchmark(lib.sigmoid_f16x8_pack, x, "f16x8_pack", ref, out)
    run_benchmark(sigmoid_triton, x, "triton", ref)

    print("-" * 105)


if __name__ == "__main__":
    bench_shape(4096, 512)
    bench_shape(4096, 1024)
    bench_shape(4096, 2048)
    bench_shape(4096, 4096)
    bench_shape(4096, 8192)
