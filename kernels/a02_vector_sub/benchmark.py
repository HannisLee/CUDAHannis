import time
from typing import Optional
import torch
from torch.utils.cpp_extension import load
torch.set_grad_enabled(False)

lib = load(
    name="vector_sub_lib",
    sources=["vector_sub.cu"],
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
            perf_func(a, b, out)
    else:
        for _ in range(warmup):
            out = perf_func(a, b)

    torch.cuda.synchronize()

    start = time.time()

    if has_out:
        for _ in range(iters):
            perf_func(a, b, out)
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
    print(" " * 40 + f"N={N}, K={K}, dtype=fp16")
    print("-" * 85)

    a = torch.randn((N, K), device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn((N, K), device="cuda", dtype=torch.float16).contiguous()
    out = torch.zeros_like(a).contiguous()

    run_benchmark(lib.vector_sub_f16, a, b, "f16", out)
    run_benchmark(lib.vector_sub_f16x2, a, b, "f16x2", out)
    run_benchmark(lib.vector_sub_f16x8, a, b, "f16x8", out)
    run_benchmark(lib.vector_sub_f16x8_pack, a, b, "f16x8_pack", out)
    from vector_sub_triton import vector_sub_triton
    run_benchmark(vector_sub_triton, a, b, "triton")
    run_benchmark(naive_vector_sub, a, b, "torch")

    print("-" * 85)


if __name__ == "__main__":
    bench_shape(4096, 512)
    bench_shape(4096, 1024)
    bench_shape(4096, 2048)
    bench_shape(4096, 4096)
    bench_shape(4096, 8192)
