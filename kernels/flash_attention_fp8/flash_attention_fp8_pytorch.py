import torch

from .flash_attention_fp8_common import (
    default_sm_scale,
    dequantize_e4m3_per_row_pytorch,
    quantize_e4m3_per_row_pytorch,
    validate_attention_inputs,
)


def flash_attention_fp8_pytorch(
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
    q_codes, q_scales = quantize_e4m3_per_row_pytorch(q)
    k_codes, k_scales = quantize_e4m3_per_row_pytorch(k)
    v_codes, v_scales = quantize_e4m3_per_row_pytorch(v)
    q_deq = dequantize_e4m3_per_row_pytorch(q_codes, q_scales, torch.float32)
    k_deq = dequantize_e4m3_per_row_pytorch(k_codes, k_scales, torch.float32)
    v_deq = dequantize_e4m3_per_row_pytorch(v_codes, v_scales, torch.float32)

    scores = torch.matmul(q_deq, k_deq.transpose(-1, -2)).mul(scale).to(torch.float16)
    seq_len = q.shape[-2]
    mask = torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, -float("inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float16)
    return torch.matmul(probs.float(), v_deq).to(torch.float16).contiguous()
