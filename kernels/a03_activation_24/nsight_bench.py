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

from activation_24_pytorch import activation_24_sparsity
from activation_24_triton_v1 import activation_24_sparsity_triton as activation_24_triton_v1
from activation_24_triton_v2 import activation_24_sparsity_triton as activation_24_triton_v2
from activation_24_triton_v6 import activation_24_sparsity_triton as activation_24_triton_v6


lib = load(
    name="activation_24_lib",
    sources=[str(CURRENT_DIR / "activation_24.cu")],
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
    perf_func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    tag: str,
    ref: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 100,
    show_all: bool = False,
):
    out = None

    for _ in range(warmup):
        out = perf_func(x)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()

    for _ in range(iters):
        out = perf_func(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    torch.cuda.cudart().cudaProfilerStop()

    assert out is not None
    mean_time = (end - start) * 1000 / iters
    out_val = out.flatten().detach().cpu().tolist()[:6]
    out_val = [f"{float(v):<12.8g}" for v in out_val]

    if ref is None:
        max_abs = 0.0
    else:
        max_abs = (out - ref).abs().max().item()

    kept_ratio = (out != 0).float().mean().item()
    print(
        f"{tag:>18}: {out_val}, "
        f"time:{mean_time:.8f}ms, max_abs:{max_abs:.3e}, kept:{kept_ratio:.4f}"
    )

    if show_all:
        print(out)

    return out, mean_time


def bench_shape(*shape: int, dtype: torch.dtype = torch.float16):
    print("-" * 105)
    print(" " * 35 + f"shape={shape}, dtype={dtype}")
    print("-" * 105)

    x = torch.randn(shape, device="cuda", dtype=dtype).contiguous()

    run_benchmark(activation_24_triton_v1, x, "triton_v1")
    

    print("-" * 105)


if __name__ == "__main__":
    bench_shape(4096, 4096)
    
