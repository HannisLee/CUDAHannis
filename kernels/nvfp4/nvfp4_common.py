import torch
import triton


FP4_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_E2M1_BOUNDARIES = torch.tensor([0.250001, 0.750001, 1.250001, 1.750001, 2.500001, 3.500001, 5.000001], dtype=torch.float32)


def validate_input(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise RuntimeError("NVFP4 quantization expects a CUDA tensor.")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"NVFP4 quantization supports float16/float32, got {x.dtype}.")
    if x.dim() == 0:
        raise ValueError("NVFP4 quantization expects at least 1 dimension.")


def padded_last_dim(last_dim: int) -> int:
    return triton.cdiv(last_dim, 16) * 16


def fp4_e2m1_codes(abs_scaled: torch.Tensor) -> torch.Tensor:
    boundaries = FP4_E2M1_BOUNDARIES.to(device=abs_scaled.device)
    return (abs_scaled.unsqueeze(-1) > boundaries).sum(dim=-1).to(torch.uint8)
