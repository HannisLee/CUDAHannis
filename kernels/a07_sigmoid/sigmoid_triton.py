import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)


def sigmoid_triton(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _sigmoid_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return y
