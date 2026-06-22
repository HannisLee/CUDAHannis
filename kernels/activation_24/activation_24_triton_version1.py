"""Activation 2:4 sparsity — Triton operator, version1.

General row + group-block kernel. Every program handles one row tile of groups,
loads four lanes per group with bounds-checked masks (so any last-dim size is
handled), and computes each lane's rank with lower-index tie breaking. Lanes
with rank < 2 keep their value; padded tail elements never win a top-2 slot.

Correctness contract (shared with the PyTorch reference): the last dimension is
treated as groups of 4 contiguous values; within each group only the 2 values
with the largest magnitude are kept, the other 2 lanes become zero, and ties in
absolute value are broken deterministically by keeping the lower-index lane.
"""

import torch
import triton
import triton.language as tl

from .activation_24_common import validate_input


@triton.jit
def _activation_24_sparsity_masked_kernel(
    x_ptr,
    out_ptr,
    last_dim: tl.constexpr,
    groups_per_row: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    group_offsets = block * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    row_base = row * last_dim
    elem0 = group_offsets * 4
    elem1 = elem0 + 1
    elem2 = elem0 + 2
    elem3 = elem0 + 3

    group_mask = group_offsets < groups_per_row
    mask0 = group_mask & (elem0 < last_dim)
    mask1 = group_mask & (elem1 < last_dim)
    mask2 = group_mask & (elem2 < last_dim)
    mask3 = group_mask & (elem3 < last_dim)

    v0 = tl.load(x_ptr + row_base + elem0, mask=mask0, other=0.0)
    v1 = tl.load(x_ptr + row_base + elem1, mask=mask1, other=0.0)
    v2 = tl.load(x_ptr + row_base + elem2, mask=mask2, other=0.0)
    v3 = tl.load(x_ptr + row_base + elem3, mask=mask3, other=0.0)

    # Padded elements map to -inf magnitude so they can never be in the top-2.
    a0 = tl.where(mask0, tl.abs(v0), -float("inf"))
    a1 = tl.where(mask1, tl.abs(v1), -float("inf"))
    a2 = tl.where(mask2, tl.abs(v2), -float("inf"))
    a3 = tl.where(mask3, tl.abs(v3), -float("inf"))

    rank0 = (a1 > a0).to(tl.int32) + (a2 > a0).to(tl.int32) + (a3 > a0).to(tl.int32)
    rank1 = (a0 >= a1).to(tl.int32) + (a2 > a1).to(tl.int32) + (a3 > a1).to(tl.int32)
    rank2 = (a0 >= a2).to(tl.int32) + (a1 >= a2).to(tl.int32) + (a3 > a2).to(tl.int32)
    rank3 = (a0 >= a3).to(tl.int32) + (a1 >= a3).to(tl.int32) + (a2 >= a3).to(tl.int32)

    tl.store(out_ptr + row_base + elem0, tl.where(rank0 < 2, v0, 0.0), mask=mask0)
    tl.store(out_ptr + row_base + elem1, tl.where(rank1 < 2, v1, 0.0), mask=mask1)
    tl.store(out_ptr + row_base + elem2, tl.where(rank2 < 2, v2, 0.0), mask=mask2)
    tl.store(out_ptr + row_base + elem3, tl.where(rank3 < 2, v3, 0.0), mask=mask3)


def activation_24_sparsity_triton_v1(x: torch.Tensor, block_groups: int = 256) -> torch.Tensor:
    """version1: general masked row + group-block kernel for any last-dim size."""
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()

    last_dim = x.shape[-1]
    out = torch.empty_like(x)
    if last_dim == 0:
        return out

    rows = x.numel() // last_dim
    groups_per_row = triton.cdiv(last_dim, 4)
    grid = (rows, triton.cdiv(groups_per_row, block_groups))
    _activation_24_sparsity_masked_kernel[grid](
        x,
        out,
        last_dim,
        groups_per_row,
        BLOCK_GROUPS=block_groups,
    )
    return out
