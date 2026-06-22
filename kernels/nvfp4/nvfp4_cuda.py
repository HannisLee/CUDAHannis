from functools import lru_cache
from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from kernels._cuda_extension import conda_compiler_paths, set_default_cuda_arch_list
from .nvfp4_common import validate_input
from .nvfp4_pytorch import nvfp4_dequantize_pytorch


KERNEL_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_nvfp4_extension():
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set and nvcc was not found. Run `source scripts/activate_env.sh` "
            "and make sure the triton-cu118 environment has cuda-nvcc installed."
        )

    sources = [
        KERNEL_DIR / "nvfp4_cuda.cpp",
        KERNEL_DIR / "nvfp4_cuda_kernel.cu",
    ]
    set_default_cuda_arch_list()
    gcc, gxx = conda_compiler_paths()
    extra_cuda_cflags = ["-O3"]
    if gcc is not None and gxx is not None:
        os.environ.setdefault("CC", str(gcc))
        os.environ.setdefault("CXX", str(gxx))
        extra_cuda_cflags.append(f"-ccbin={gxx}")

    return load(
        name="nvfp4_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True,
    )


def nvfp4_quantize_cuda(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()

    module = _load_nvfp4_extension()
    return module.quantize(x)


def nvfp4_quantize_dequantize_cuda(x: torch.Tensor) -> torch.Tensor:
    packed, scales = nvfp4_quantize_cuda(x)
    return nvfp4_dequantize_pytorch(packed, scales, x.shape[-1], out_dtype=x.dtype)
