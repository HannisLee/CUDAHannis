import torch
import triton
import triton.language as tl


@triton.jit
def _max_stage_kernel(
    x_ptr,
    partial_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    partial = tl.max(x, axis=0)
    tl.store(partial_ptr + pid, partial)


def max_triton(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    assert x.is_contiguous()

    n_elements = x.numel()
    assert n_elements > 0

    current = x.reshape(-1)
    current_n = n_elements

    while current_n > 1:
        n_blocks = triton.cdiv(current_n, block_size)
        partial = torch.empty((n_blocks,), device=x.device, dtype=x.dtype)
        _max_stage_kernel[(n_blocks,)](
            current,
            partial,
            current_n,
            BLOCK_SIZE=block_size,
        )
        current = partial
        current_n = n_blocks

    return current.reshape(1)
