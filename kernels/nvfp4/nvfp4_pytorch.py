import torch
import torch.nn.functional as F

from .nvfp4_common import FP4_E2M1_VALUES, fp4_e2m1_codes, padded_last_dim, validate_input


def nvfp4_quantize_pytorch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()

    last_dim = x.shape[-1]
    padded = padded_last_dim(last_dim)
    pad = padded - last_dim
    padded_x = F.pad(x, (0, pad)) if pad else x
    grouped = padded_x.reshape(-1, padded // 16, 16)
    grouped_float = grouped.float()

    offsets = torch.arange(padded, device=x.device).reshape(1, -1, 16)
    valid = offsets < last_dim
    abs_values = torch.where(valid, grouped_float.abs(), torch.zeros_like(grouped_float))
    max_abs = abs_values.amax(dim=-1)
    scales = torch.where(max_abs > 0, max_abs / 6.0, torch.ones_like(max_abs)).to(torch.float32)

    normalized = torch.clamp(abs_values / scales.unsqueeze(-1), max=6.0)
    magnitude = fp4_e2m1_codes(normalized)
    sign = (grouped < 0).to(torch.uint8) << 3
    codes = torch.where(valid, magnitude | sign, torch.zeros_like(magnitude))
    pairs = codes.reshape(-1, padded // 16, 8, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)

    return packed.reshape(*x.shape[:-1], padded // 2).contiguous(), scales.reshape(
        *x.shape[:-1], padded // 16
    ).contiguous()


def nvfp4_dequantize_pytorch(
    packed: torch.Tensor,
    scales: torch.Tensor,
    original_last_dim: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed NVFP4 tensor must be uint8, got {packed.dtype}.")
    if scales.dtype != torch.float32:
        raise TypeError(f"NVFP4 scales must be float32 in this educational implementation, got {scales.dtype}.")
    if packed.shape[:-1] != scales.shape[:-1]:
        raise ValueError("packed and scales prefix shapes must match.")

    blocks_per_row = scales.shape[-1]
    expected_packed_last_dim = blocks_per_row * 8
    if packed.shape[-1] != expected_packed_last_dim:
        raise ValueError(
            f"packed last dim must be blocks_per_row * 8, got {packed.shape[-1]} and {blocks_per_row} blocks."
        )

    bytes_flat = packed.reshape(-1, expected_packed_last_dim)
    low = bytes_flat & 0x0F
    high = (bytes_flat >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(-1, blocks_per_row, 16)
    magnitudes = codes & 0x07
    signs = (codes & 0x08) != 0

    values_lut = FP4_E2M1_VALUES.to(device=packed.device)
    values = values_lut[magnitudes.long()]
    values = torch.where(signs, -values, values)
    values = values * scales.reshape(-1, blocks_per_row).unsqueeze(-1)
    values = values.reshape(*packed.shape[:-1], blocks_per_row * 16)
    return values[..., :original_last_dim].to(out_dtype).contiguous()


def nvfp4_quantize_dequantize_pytorch(x: torch.Tensor) -> torch.Tensor:
    packed, scales = nvfp4_quantize_pytorch(x)
    return nvfp4_dequantize_pytorch(packed, scales, x.shape[-1], out_dtype=x.dtype)
