from functools import lru_cache
from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from kernels._cuda_extension import conda_compiler_paths, set_default_cuda_arch_list
from .rms_norm_common import (
    EPSILON,
    SUPPORTED_K,
    VARIANT_FUNC,
    resolve_variant,
    validate_input,
)


KERNEL_DIR = Path(__file__).resolve().parent

# rms_norm_f16_f32.cu is a single self-contained translation unit: it holds the
# device kernels, the host dispatch wrappers and the PYBIND11_MODULE binding.
# Unlike the .cpp + .cu split used elsewhere in this repo, splitting it would
# require a shared header for the templated kernels, so we JIT-load it directly.
KERNEL_SOURCE = KERNEL_DIR / "rms_norm_f16_f32.cu"


@lru_cache(maxsize=1)
def _load_rms_norm_extension():
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set and nvcc was not found. Run `source scripts/activate_env.sh` "
            "and make sure the triton-cu118 environment has cuda-nvcc installed."
        )

    set_default_cuda_arch_list()
    gcc, gxx = conda_compiler_paths()
    # rms_norm_f16_f32.cu uses half/half2 arithmetic operators (*, /, +=). Torch's
    # cpp_extension disables them by default (COMMON_NVCC_FLAGS defines
    # __CUDA_NO_HALF_OPERATORS__ etc.); -U comes after those flags, so it wins and
    # re-enables the operators. Safe here because the kernels use raw cuda half,
    # never torch's c10::Half, so there is no conversion ambiguity.
    extra_cuda_cflags = [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    if gcc is not None and gxx is not None:
        os.environ.setdefault("CC", str(gcc))
        os.environ.setdefault("CXX", str(gxx))
        extra_cuda_cflags.append(f"-ccbin={gxx}")

    return load(
        name="rms_norm_ext",
        sources=[str(KERNEL_SOURCE)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True,
    )


def rms_norm_cuda(
    x: torch.Tensor,
    g: float,
    eps: float = EPSILON,
    variant: str = "auto",
) -> torch.Tensor:
    """RMSNorm via the CUDA extension.

    ``variant`` selects one of the 9 kernels in rms_norm_f16_f32.cu ("auto" picks
    the fastest for the input dtype). The CUDA kernels hardcode eps=1e-5, so eps
    must equal EPSILON; pytorch/triton honour a custom eps instead.
    """
    if eps != EPSILON:
        raise ValueError(
            f"The CUDA kernels hardcode eps={EPSILON}; pass eps={EPSILON} "
            f"(or use the pytorch/triton backend for a custom eps)."
        )
    validate_input(x, g)
    if not x.is_contiguous():
        x = x.contiguous()

    n, k = x.shape
    name = resolve_variant(variant, x.dtype, k)
    module = _load_rms_norm_extension()
    y = torch.empty_like(x)
    getattr(module, VARIANT_FUNC[name])(x, y, float(g))
    return y


def cuda_variants(x: torch.Tensor, k: int | None = None) -> list[str]:
    """Convenience accessor for tests: which variant tags this tensor can use."""
    from .rms_norm_common import variants_for

    return variants_for(x.dtype, k if k is not None else x.shape[-1])


__all__ = ["rms_norm_cuda", "cuda_variants", "SUPPORTED_K"]
