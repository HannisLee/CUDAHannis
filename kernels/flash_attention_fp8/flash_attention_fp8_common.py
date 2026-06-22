from __future__ import annotations

import math

import torch


FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 2.0**-6
FP8_E4M3_SUBNORMAL_SCALE = 2.0**9


def validate_attention_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise RuntimeError("flash_attention_fp8 expects CUDA tensors.")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must have the same shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}.")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError(f"flash_attention_fp8 supports float16 inputs, got {q.dtype}, {k.dtype}, {v.dtype}.")
    if q.dim() != 4:
        raise ValueError(f"q, k, v must have shape (B, H, S, D), got {tuple(q.shape)}.")
    if q.size(-1) not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {q.size(-1)}.")
    if q.size(-2) <= 0:
        raise ValueError("sequence length must be non-empty.")


def default_sm_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def e4m3_encode_pytorch(values: torch.Tensor) -> torch.Tensor:
    """Encode non-negative float values to finite E4M3 codes without the sign bit."""
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=FP8_E4M3_MAX, neginf=0.0)
    values = torch.clamp(values, min=0.0, max=FP8_E4M3_MAX)

    subnormal = values < FP8_E4M3_MIN_NORMAL
    sub_mant = torch.floor(values * FP8_E4M3_SUBNORMAL_SCALE + 0.5).to(torch.int32).clamp_(0, 7)

    safe = torch.clamp(values, min=FP8_E4M3_MIN_NORMAL)
    exp_unbiased = torch.floor(torch.log2(safe)).to(torch.int32)
    base = torch.pow(2.0, exp_unbiased.float())
    mant = torch.floor((safe / base - 1.0) * 8.0 + 0.5).to(torch.int32)
    carry = mant >= 8
    exp_unbiased = exp_unbiased + carry.to(torch.int32)
    mant = torch.where(carry, torch.zeros_like(mant), mant)

    exp_biased = exp_unbiased + 7
    overflow = exp_biased > 15
    exp_biased = exp_biased.clamp_(1, 15)
    mant = torch.where(overflow, torch.full_like(mant, 6), mant.clamp(0, 7))
    mant = torch.where((exp_biased == 15) & (mant > 6), torch.full_like(mant, 6), mant)

    normal_code = (exp_biased << 3) | mant
    code = torch.where(subnormal, sub_mant, normal_code)
    return code.to(torch.uint8)


def e4m3_decode_pytorch(codes: torch.Tensor) -> torch.Tensor:
    codes_i = codes.to(torch.int32)
    sign = (codes_i & 0x80) != 0
    mag = codes_i & 0x7F
    exp = (mag >> 3) & 0x0F
    mant = mag & 0x07
    mant = torch.where((exp == 15) & (mant == 7), torch.full_like(mant, 6), mant)

    sub = mant.float() * (2.0**-9)
    normal = torch.pow(2.0, exp.float() - 7.0) * (1.0 + mant.float() / 8.0)
    values = torch.where(exp == 0, sub, normal)
    values = torch.where(sign, -values, values)
    return values


def quantize_e4m3_per_row_pytorch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()

    rows = x.reshape(-1, x.shape[-1]).float()
    max_abs = rows.abs().amax(dim=-1)
    scales = torch.where(max_abs > 0.0, max_abs / FP8_E4M3_MAX, torch.ones_like(max_abs)).float()
    normalized = torch.clamp(rows.abs() / scales.unsqueeze(-1), max=FP8_E4M3_MAX)
    magnitude = e4m3_encode_pytorch(normalized)
    sign = ((rows < 0.0).to(torch.uint8) << 7)
    codes = magnitude | sign
    return codes.reshape_as(x).contiguous(), scales.reshape(*x.shape[:-1]).contiguous()


def dequantize_e4m3_per_row_pytorch(codes: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    values = e4m3_decode_pytorch(codes) * scales.unsqueeze(-1).float()
    return values.to(out_dtype).contiguous()
