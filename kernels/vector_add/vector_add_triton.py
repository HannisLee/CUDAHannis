import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise RuntimeError("triton_add expects CUDA tensors. Please move inputs to GPU first.")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.dtype != y.dtype:
        raise ValueError(f"x and y must have the same dtype, got {x.dtype} and {y.dtype}")
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out
