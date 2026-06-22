from .flash_attention_fp8_cuda import flash_attention_fp8_cuda
from .flash_attention_fp8_pytorch import flash_attention_fp8_pytorch
from .flash_attention_fp8_triton import flash_attention_fp8_triton


def flash_attention_fp8(q, k, v, backend: str = "triton", sm_scale: float | None = None):
    if backend == "pytorch":
        return flash_attention_fp8_pytorch(q, k, v, sm_scale=sm_scale)
    if backend == "triton":
        return flash_attention_fp8_triton(q, k, v, sm_scale=sm_scale)
    if backend == "cuda":
        return flash_attention_fp8_cuda(q, k, v, sm_scale=sm_scale)
    raise ValueError(f"Unknown backend {backend!r}. Expected 'pytorch', 'triton', or 'cuda'.")


__all__ = [
    "flash_attention_fp8",
    "flash_attention_fp8_cuda",
    "flash_attention_fp8_pytorch",
    "flash_attention_fp8_triton",
]
