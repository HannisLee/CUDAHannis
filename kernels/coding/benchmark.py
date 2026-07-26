import time
from typing import Optional
import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

torch.set_grad_enabled(False)

lib = load(
    name="cudalib",
    sources=["coding.cu"],
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
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
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
            perf_func(a, out)
    else:
        for _ in range(warmup):
            out = perf_func(a, b)

    torch.cuda.synchronize()

    start = time.time()

    if has_out:
        for _ in range(iters):
            perf_func(a, out)
    else:
        for _ in range(iters):
            out = perf_func(a, b)

    torch.cuda.synchronize()

    end = time.time()

    total_time = (end - start) * 1000
    mean_time = total_time / iters

    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(float(v), 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]

    print(f"{out_info:>20}: {out_val}, time:{mean_time:.8f}ms")

    if show_all:
        print(out)

    return out, mean_time

def naive_vector_sub(a: torch.Tensor, b: torch.Tensor):
    return a - b


def bench_shape(N: int, K: int):
    print("-" * 85)
    print(" " * 40 + f"N={N}, K={K}, dtype=fp32")
    print("-" * 85)

    a = torch.randn((N, K), device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn((N, K), device="cuda", dtype=torch.float32).contiguous()
    out = torch.zeros_like(a).contiguous()

    run_benchmark(lib.coding1, a, b, "cuda_out", out)
    


    print("-" * 85)


if __name__ == "__main__":
    shapes = [(4096, 512), (4096, 1024), (4096, 2048), (4096, 4096), (4096, 8192)]
    for N, K in shapes:
        bench_shape(N, K)
