import torch

from .rms_norm_common import EPSILON, validate_input


def rms_norm_pytorch(x: torch.Tensor, g: float, eps: float = EPSILON) -> torch.Tensor:
    """Reference RMSNorm: y = x * rsqrt(mean(x^2) + eps) * g.

    Accumulates sum-of-squares in float32 (matching the fp32-accumulating CUDA
    variants) regardless of input dtype, then casts back to the input dtype.
    The scale ``g`` is a scalar applied to every element, matching the CUDA
    kernels' scalar ``g`` argument.
    """
    validate_input(x, g)
    if not x.is_contiguous():
        x = x.contiguous()

    xf = x.float()
    variance = xf.pow(2).mean(dim=-1)
    rrms = torch.rsqrt(variance + eps)
    out = (xf * rrms.unsqueeze(-1) * float(g)).to(x.dtype)
    return out.contiguous()
