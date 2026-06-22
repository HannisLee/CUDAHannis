"""Activation 2:4 sparsity — CUDA operator, versions 1-4.

JIT-builds one extension that links the four per-version CUDA kernels and
exposes four Python entry points, mirroring the Triton ``activation_24_triton``
version1-4 modules:

* ``activation_24_sparsity_cuda_v1`` — masked group kernel (general, any shape).
* ``activation_24_sparsity_cuda_v2`` — aligned mask-free fast path.
* ``activation_24_sparsity_cuda_v3`` — vectorized contiguous load/store.
* ``activation_24_sparsity_cuda_v4`` — vectorized + branchless pairwise top-2.

Each version falls back to its own masked group kernel for irregular last-dim
sizes (last_dim not a multiple of 4), just like the Triton versions.
"""

from functools import lru_cache
from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from kernels._cuda_extension import conda_compiler_paths, set_default_cuda_arch_list
from .activation_24_common import validate_input


KERNEL_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_activation_24_sparsity_versioned_extension():
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set and nvcc was not found. Run `source scripts/activate_env.sh` "
            "and make sure the triton-cu118 environment has cuda-nvcc installed."
        )

    sources = [
        KERNEL_DIR / "activation_24_cuda_version1_4.cpp",
        KERNEL_DIR / "activation_24_cuda_kernel_version1.cu",
        KERNEL_DIR / "activation_24_cuda_kernel_version2.cu",
        KERNEL_DIR / "activation_24_cuda_kernel_version3.cu",
        KERNEL_DIR / "activation_24_cuda_kernel_version4.cu",
    ]
    set_default_cuda_arch_list()
    gcc, gxx = conda_compiler_paths()
    extra_cuda_cflags = ["-O3"]
    if gcc is not None and gxx is not None:
        os.environ.setdefault("CC", str(gcc))
        os.environ.setdefault("CXX", str(gxx))
        extra_cuda_cflags.append(f"-ccbin={gxx}")

    return load(
        name="activation_24_sparsity_versioned_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True,
    )


def _forward(version: str, x: torch.Tensor) -> torch.Tensor:
    module = _load_activation_24_sparsity_versioned_extension()
    return getattr(module, f"forward_{version}")(x)


def activation_24_sparsity_cuda_v1(x: torch.Tensor) -> torch.Tensor:
    """version1: masked group kernel (general, any last-dim size)."""
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()
    return _forward("v1", x)


def activation_24_sparsity_cuda_v2(x: torch.Tensor) -> torch.Tensor:
    """version2: aligned mask-free fast path, masked fallback otherwise."""
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()
    return _forward("v2", x)


def activation_24_sparsity_cuda_v3(x: torch.Tensor) -> torch.Tensor:
    """version3: vectorized contiguous load/store, masked fallback otherwise."""
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()
    return _forward("v3", x)


def activation_24_sparsity_cuda_v4(x: torch.Tensor) -> torch.Tensor:
    """version4: vectorized + branchless pairwise top-2, masked fallback otherwise."""
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()
    return _forward("v4", x)
