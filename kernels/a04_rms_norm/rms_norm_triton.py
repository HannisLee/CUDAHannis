import triton
import triton.language as tl
import torch

from .rms_norm_common import EPSILON, validate_input


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    y_ptr,
    g: tl.constexpr,
    eps: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per row; matches the CUDA kernels' one-block-per-row layout.
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    x_row = x_ptr + row * K
    y_row = y_ptr + row * K

    x = tl.load(x_row + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    variance = tl.sum(xf * xf, axis=0) / K
    rrms = tl.rsqrt(variance + eps)
    y = (xf * rrms * g).to(x.dtype)
    tl.store(y_row + cols, y, mask=mask)


def rms_norm_triton(x: torch.Tensor, g: float, eps: float = EPSILON) -> torch.Tensor:
    validate_input(x, g)
    if not x.is_contiguous():
        x = x.contiguous()

    n_rows, k = x.shape
    y = torch.empty_like(x)
    block_k = triton.next_power_of_2(k)
    num_warps = min(16, max(4, block_k // 256))
    _rms_norm_kernel[(n_rows,)](
        x,
        y,
        float(g),
        eps,
        k,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return y
