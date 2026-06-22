from functools import lru_cache
from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from kernels._cuda_extension import conda_compiler_paths, set_default_cuda_arch_list


KERNEL_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_cuda_add_extension():
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set and nvcc was not found. Run `source scripts/activate_env.sh` "
            "and make sure the triton-cu118 environment has cuda-nvcc installed."
        )

    sources = [
        KERNEL_DIR / "vector_add_cuda.cpp",
        KERNEL_DIR / "vector_add_cuda_kernel.cu",
    ]
    set_default_cuda_arch_list()
    gcc, gxx = conda_compiler_paths()
    extra_cuda_cflags = ["-O3"]
    if gcc is not None and gxx is not None:
        os.environ.setdefault("CC", str(gcc))
        os.environ.setdefault("CXX", str(gxx))
        extra_cuda_cflags.append(f"-ccbin={gxx}")

    return load(
        name="cuda_add_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True,
    )


def cuda_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise RuntimeError("cuda_add expects CUDA tensors. Please move inputs to GPU first.")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.dtype != y.dtype:
        raise ValueError(f"x and y must have the same dtype, got {x.dtype} and {y.dtype}")
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    module = _load_cuda_add_extension()
    return module.forward(x, y)
