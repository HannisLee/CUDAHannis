import torch
import triton
import triton.language as tl

from .nvfp4_common import padded_last_dim, validate_input
from .nvfp4_pytorch import nvfp4_dequantize_pytorch


@triton.jit
def _nvfp4_quantize_kernel(
    x_ptr,
    packed_ptr,
    scales_ptr,
    last_dim: tl.constexpr,
    blocks_per_row: tl.constexpr,
):
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    offsets = block * 16 + tl.arange(0, 16)
    mask = offsets < last_dim
    row_base = row * last_dim
    values = tl.load(x_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    abs_values = tl.abs(values)
    max_abs = tl.max(abs_values, axis=0)
    scale = tl.where(max_abs > 0.0, max_abs / 6.0, 1.0)
    normalized = tl.minimum(abs_values / scale, 6.0)

    pair_offsets = tl.arange(0, 8)
    even_offsets = block * 16 + pair_offsets * 2
    odd_offsets = even_offsets + 1
    even_mask = even_offsets < last_dim
    odd_mask = odd_offsets < last_dim
    even_values = tl.load(x_ptr + row_base + even_offsets, mask=even_mask, other=0.0).to(tl.float32)
    odd_values = tl.load(x_ptr + row_base + odd_offsets, mask=odd_mask, other=0.0).to(tl.float32)
    even_normalized = tl.minimum(tl.abs(even_values) / scale, 6.0)
    odd_normalized = tl.minimum(tl.abs(odd_values) / scale, 6.0)

    low = tl.full((8,), 0, tl.uint8)
    low += (even_normalized > 0.250001).to(tl.uint8)
    low += (even_normalized > 0.750001).to(tl.uint8)
    low += (even_normalized > 1.250001).to(tl.uint8)
    low += (even_normalized > 1.750001).to(tl.uint8)
    low += (even_normalized > 2.500001).to(tl.uint8)
    low += (even_normalized > 3.500001).to(tl.uint8)
    low += (even_normalized > 5.000001).to(tl.uint8)
    low = tl.where(even_mask, low | ((even_values < 0.0).to(tl.uint8) << 3), 0)

    high = tl.full((8,), 0, tl.uint8)
    high += (odd_normalized > 0.250001).to(tl.uint8)
    high += (odd_normalized > 0.750001).to(tl.uint8)
    high += (odd_normalized > 1.250001).to(tl.uint8)
    high += (odd_normalized > 1.750001).to(tl.uint8)
    high += (odd_normalized > 2.500001).to(tl.uint8)
    high += (odd_normalized > 3.500001).to(tl.uint8)
    high += (odd_normalized > 5.000001).to(tl.uint8)
    high = tl.where(odd_mask, high | ((odd_values < 0.0).to(tl.uint8) << 3), 0)

    packed = low | (high << 4)
    packed_base = row * blocks_per_row * 8 + block * 8
    scale_base = row * blocks_per_row + block
    tl.store(packed_ptr + packed_base + pair_offsets, packed)
    tl.store(scales_ptr + scale_base, scale)


def nvfp4_quantize_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()

    last_dim = x.shape[-1]
    padded = padded_last_dim(last_dim)
    rows = x.numel() // last_dim
    blocks_per_row = padded // 16
    packed = torch.empty((*x.shape[:-1], padded // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((*x.shape[:-1], blocks_per_row), device=x.device, dtype=torch.float32)
    _nvfp4_quantize_kernel[(rows, blocks_per_row)](x, packed, scales, last_dim, blocks_per_row)
    return packed, scales


def nvfp4_quantize_dequantize_triton(x: torch.Tensor) -> torch.Tensor:
    packed, scales = nvfp4_quantize_triton(x)
    return nvfp4_dequantize_pytorch(packed, scales, x.shape[-1], out_dtype=x.dtype)
