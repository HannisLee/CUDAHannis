import torch
import triton
import triton.language as tl

from .flash_attention_fp8_common import default_sm_scale, validate_attention_inputs


@triton.jit
def _e4m3_encode_magnitude(values):
    values = tl.minimum(tl.maximum(values, 0.0), 448.0)
    subnormal = values < 0.015625
    sub_mant = tl.minimum(tl.floor(values * 512.0 + 0.5), 7.0).to(tl.uint32)

    safe = tl.maximum(values, 0.015625)
    exp_unbiased_f = tl.floor(tl.log2(safe))
    base = tl.exp2(exp_unbiased_f)
    mant_f = tl.floor((safe / base - 1.0) * 8.0 + 0.5)
    carry = mant_f >= 8.0
    exp_unbiased = exp_unbiased_f.to(tl.int32) + carry.to(tl.int32)
    mant = tl.where(carry, 0, mant_f.to(tl.int32))
    exp_biased = exp_unbiased + 7
    overflow = exp_biased > 15
    exp_biased = tl.minimum(tl.maximum(exp_biased, 1), 15)
    mant = tl.where(overflow, 6, tl.minimum(tl.maximum(mant, 0), 7))
    mant = tl.where((exp_biased == 15) & (mant > 6), 6, mant)
    normal_code = ((exp_biased.to(tl.uint32) << 3) | mant.to(tl.uint32))
    return tl.where(subnormal, sub_mant, normal_code).to(tl.uint8)


@triton.jit
def _e4m3_decode(codes):
    code = codes.to(tl.uint32)
    negative = (code & 0x80) != 0
    mag = code & 0x7F
    exp = (mag >> 3) & 0x0F
    mant = mag & 0x07
    mant = tl.where((exp == 15) & (mant == 7), 6, mant)
    sub = mant.to(tl.float32) * 0.001953125
    normal = tl.exp2(exp.to(tl.float32) - 7.0) * (1.0 + mant.to(tl.float32) * 0.125)
    values = tl.where(exp == 0, sub, normal)
    return tl.where(negative, -values, values)


@triton.jit
def _quantize_e4m3_per_row_kernel(
    x_ptr,
    codes_ptr,
    scales_ptr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < head_dim
    values = tl.load(x_ptr + row * head_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    abs_values = tl.abs(values)
    max_abs = tl.max(tl.where(mask, abs_values, 0.0), axis=0)
    scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    normalized = tl.minimum(abs_values / scale, 448.0)
    magnitude = _e4m3_encode_magnitude(normalized)
    sign = (values < 0.0).to(tl.uint8) << 7
    codes = magnitude | sign
    tl.store(codes_ptr + row * head_dim + offsets, codes, mask=mask)
    tl.store(scales_ptr + row, scale)


@triton.jit
def _flash_attention_fp8_forward_kernel(
    q_codes_ptr,
    k_codes_ptr,
    v_codes_ptr,
    q_scales_ptr,
    k_scales_ptr,
    v_scales_ptr,
    out_ptr,
    sm_scale,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    bh_data_base = pid_bh * seq_len * head_dim
    bh_scale_base = pid_bh * seq_len

    q_code = tl.load(
        q_codes_ptr + bh_data_base + offs_m[:, None] * head_dim + offs_d[None, :],
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0,
    )
    q_scale = tl.load(q_scales_ptr + bh_scale_base + offs_m, mask=offs_m < seq_len, other=1.0).to(tl.float32)
    q = _e4m3_decode(q_code) * q_scale[:, None]

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        n = start_n + offs_n
        k_code = tl.load(
            k_codes_ptr + bh_data_base + n[:, None] * head_dim + offs_d[None, :],
            mask=(n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0,
        )
        v_code = tl.load(
            v_codes_ptr + bh_data_base + n[:, None] * head_dim + offs_d[None, :],
            mask=(n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0,
        )
        k_scale = tl.load(k_scales_ptr + bh_scale_base + n, mask=n < seq_len, other=1.0).to(tl.float32)
        v_scale = tl.load(v_scales_ptr + bh_scale_base + n, mask=n < seq_len, other=1.0).to(tl.float32)
        k = _e4m3_decode(k_code) * k_scale[:, None]
        v = _e4m3_decode(v_code) * v_scale[:, None]

        scores = tl.dot(q, tl.trans(k), input_precision="tf32") * sm_scale
        causal = n[None, :] <= offs_m[:, None]
        valid = (offs_m[:, None] < seq_len) & (n[None, :] < seq_len) & causal
        scores = tl.where(valid, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v, input_precision="tf32")
        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]
    tl.store(
        out_ptr + bh_data_base + offs_m[:, None] * head_dim + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
    )


def _quantize_e4m3_per_row_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.numel() // x.shape[-1]
    head_dim = x.shape[-1]
    codes = torch.empty_like(x, dtype=torch.uint8)
    scales = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    _quantize_e4m3_per_row_kernel[(rows,)](
        x,
        codes,
        scales,
        head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return codes, scales


def flash_attention_fp8_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    validate_attention_inputs(q, k, v)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    scale = default_sm_scale(q.shape[-1]) if sm_scale is None else float(sm_scale)
    q_codes, q_scales = _quantize_e4m3_per_row_triton(q)
    k_codes, k_scales = _quantize_e4m3_per_row_triton(k)
    v_codes, v_scales = _quantize_e4m3_per_row_triton(v)

    batch, heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    block_m = 16
    block_n = 64
    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    _flash_attention_fp8_forward_kernel[grid](
        q_codes,
        k_codes,
        v_codes,
        q_scales,
        k_scales,
        v_scales,
        out,
        scale,
        seq_len,
        head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=3,
    )
    return out
