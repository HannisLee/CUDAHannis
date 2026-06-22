"""Activation 2:4 sparsity — Triton operator, version3.

Adds a contiguous-element fast path for fully aligned tensors (last_dim a
multiple of 4 and numel divisible by the block size). Each program owns a run
of contiguous elements; the current lane comes from the global offset
(``offset & 3``) and the whole 4-element group is loaded by every member (a
redundant neighbour load traded for fully coalesced reads/stores). Block sizes
are 128 for float32 and 256 for float16. The version1/version2 masked and
full-block kernels remain for irregular shapes.

Correctness contract is unchanged: per group of 4, keep the 2 largest-magnitude
values, ties broken toward the lower-index lane.
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


@triton.jit
def _activation_24_sparsity_full_kernel(
    x_ptr,
    out_ptr,
    last_dim: tl.constexpr,
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

    v0 = tl.load(x_ptr + row_base + elem0)
    v1 = tl.load(x_ptr + row_base + elem1)
    v2 = tl.load(x_ptr + row_base + elem2)
    v3 = tl.load(x_ptr + row_base + elem3)

    a0 = tl.abs(v0)
    a1 = tl.abs(v1)
    a2 = tl.abs(v2)
    a3 = tl.abs(v3)

    rank0 = (a1 > a0).to(tl.int32) + (a2 > a0).to(tl.int32) + (a3 > a0).to(tl.int32)
    rank1 = (a0 >= a1).to(tl.int32) + (a2 > a1).to(tl.int32) + (a3 > a1).to(tl.int32)
    rank2 = (a0 >= a2).to(tl.int32) + (a1 >= a2).to(tl.int32) + (a3 > a2).to(tl.int32)
    rank3 = (a0 >= a3).to(tl.int32) + (a1 >= a3).to(tl.int32) + (a2 >= a3).to(tl.int32)

    tl.store(out_ptr + row_base + elem0, tl.where(rank0 < 2, v0, 0.0))
    tl.store(out_ptr + row_base + elem1, tl.where(rank1 < 2, v1, 0.0))
    tl.store(out_ptr + row_base + elem2, tl.where(rank2 < 2, v2, 0.0))
    tl.store(out_ptr + row_base + elem3, tl.where(rank3 < 2, v3, 0.0))


@triton.jit
def _activation_24_sparsity_contiguous_kernel(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lane = offsets & 3
    group_base = offsets - lane

    v = tl.load(x_ptr + offsets)
    v0 = tl.load(x_ptr + group_base)
    v1 = tl.load(x_ptr + group_base + 1)
    v2 = tl.load(x_ptr + group_base + 2)
    v3 = tl.load(x_ptr + group_base + 3)

    a0 = tl.abs(v0)
    a1 = tl.abs(v1)
    a2 = tl.abs(v2)
    a3 = tl.abs(v3)

    rank0 = (a1 > a0).to(tl.int32) + (a2 > a0).to(tl.int32) + (a3 > a0).to(tl.int32)
    rank1 = (a0 >= a1).to(tl.int32) + (a2 > a1).to(tl.int32) + (a3 > a1).to(tl.int32)
    rank2 = (a0 >= a2).to(tl.int32) + (a1 >= a2).to(tl.int32) + (a3 > a2).to(tl.int32)
    rank3 = (a0 >= a3).to(tl.int32) + (a1 >= a3).to(tl.int32) + (a2 >= a3).to(tl.int32)

    rank = tl.where(
        lane == 0, rank0, tl.where(lane == 1, rank1, tl.where(lane == 2, rank2, rank3))
    )

    tl.store(out_ptr + offsets, tl.where(rank < 2, v, 0.0))


def activation_24_sparsity_triton_v3(x: torch.Tensor, block_groups: int = 64) -> torch.Tensor:
    """version3: adds a contiguous-element fast path (128 fp32 / 256 fp16)."""
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()

    last_dim = x.shape[-1]
    out = torch.empty_like(x)
    if last_dim == 0:
        return out

    contiguous_block_size = 256 if x.dtype == torch.float16 else 128
    if last_dim % 4 == 0 and x.numel() % contiguous_block_size == 0:
        _activation_24_sparsity_contiguous_kernel[(x.numel() // contiguous_block_size,)](
            x,
            out,
            BLOCK_SIZE=contiguous_block_size,
        )
        return out

    rows = x.numel() // last_dim
    groups_per_row = triton.cdiv(last_dim, 4)
    grid = (rows, triton.cdiv(groups_per_row, block_groups))
    if last_dim % 4 == 0 and groups_per_row % block_groups == 0:
        _activation_24_sparsity_full_kernel[grid](
            x,
            out,
            last_dim,
            BLOCK_GROUPS=block_groups,
            num_warps=2,
        )
    else:
        _activation_24_sparsity_masked_kernel[grid](
            x,
            out,
            last_dim,
            groups_per_row,
            BLOCK_GROUPS=block_groups,
        )
    return out
