from .rms_norm_common import EPSILON, ALL_VARIANTS, SUPPORTED_K
from .rms_norm_cuda import rms_norm_cuda
from .rms_norm_pytorch import rms_norm_pytorch
from .rms_norm_triton import rms_norm_triton


def rms_norm(x, g, eps=EPSILON, backend: str = "triton", variant: str = "auto"):
    """Dispatch to a backend: 'pytorch', 'triton', or 'cuda'.

    'cuda' forwards ``variant`` to select one of the kernels in
    rms_norm_f16_f32.cu ('auto' picks the fastest for the dtype).
    """
    if backend == "pytorch":
        return rms_norm_pytorch(x, g, eps)
    if backend == "triton":
        return rms_norm_triton(x, g, eps)
    if backend == "cuda":
        return rms_norm_cuda(x, g, eps, variant=variant)
    raise ValueError(f"Unknown backend {backend!r}. Expected 'pytorch', 'triton', or 'cuda'.")


__all__ = [
    "ALL_VARIANTS",
    "EPSILON",
    "SUPPORTED_K",
    "rms_norm",
    "rms_norm_cuda",
    "rms_norm_pytorch",
    "rms_norm_triton",
]
