from functools import lru_cache
from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from kernels._cuda_extension import conda_compiler_paths, set_default_cuda_arch_list
from .activation_24_common import validate_input


KERNEL_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_activation_24_sparsity_extension():
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set and nvcc was not found. Run `source scripts/activate_env.sh` "
            "and make sure the triton-cu118 environment has cuda-nvcc installed."
        )

    sources = [
        KERNEL_DIR / "activation_24_cuda.cpp",
        KERNEL_DIR / "activation_24_cuda_kernel.cu",
    ]
    set_default_cuda_arch_list()
    gcc, gxx = conda_compiler_paths()
    extra_cuda_cflags = ["-O3"]
    if gcc is not None and gxx is not None:
        os.environ.setdefault("CC", str(gcc))
        os.environ.setdefault("CXX", str(gxx))
        extra_cuda_cflags.append(f"-ccbin={gxx}")

    return load(
        name="activation_24_sparsity_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True,
    )


def activation_24_sparsity_cuda(x: torch.Tensor) -> torch.Tensor:
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()

    module = _load_activation_24_sparsity_extension()
    return module.forward(x)
