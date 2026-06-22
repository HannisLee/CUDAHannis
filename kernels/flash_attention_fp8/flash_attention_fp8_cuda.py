from functools import lru_cache
from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from kernels._cuda_extension import conda_compiler_paths, set_default_cuda_arch_list
from .flash_attention_fp8_common import default_sm_scale, validate_attention_inputs


KERNEL_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_flash_attention_fp8_extension():
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set and nvcc was not found. Run `source scripts/activate_env.sh` "
            "and make sure the triton-cu118 environment has cuda-nvcc installed."
        )

    sources = [
        KERNEL_DIR / "flash_attention_fp8_cuda.cpp",
        KERNEL_DIR / "flash_attention_fp8_cuda_kernel.cu",
    ]
    set_default_cuda_arch_list()
    gcc, gxx = conda_compiler_paths()
    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    if gcc is not None and gxx is not None:
        os.environ.setdefault("CC", str(gcc))
        os.environ.setdefault("CXX", str(gxx))
        extra_cuda_cflags.append(f"-ccbin={gxx}")

    return load(
        name="flash_attention_fp8_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True,
    )


def flash_attention_fp8_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    validate_attention_inputs(q, k, v)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    scale = default_sm_scale(q.shape[-1]) if sm_scale is None else float(sm_scale)
    module = _load_flash_attention_fp8_extension()
    return module.forward(q, k, v, scale)
