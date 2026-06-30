import torch
import torch.nn.functional as F


def softmax_pytorch(x: torch.Tensor) -> torch.Tensor:
    """Per-token (row-wise) softmax reference.

    与 CUDA kernel 中的 "safe softmax" 语义保持一致：归约时上采样到 fp32，
    以提升数值稳定性，最后再转回输入 dtype。
    """
    if not x.is_contiguous():
        x = x.contiguous()
    return F.softmax(x.float(), dim=-1).to(x.dtype)
