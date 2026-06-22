import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.flash_attention_fp8 import flash_attention_fp8_cuda, flash_attention_fp8_pytorch, flash_attention_fp8_triton


def time_cuda(fn, warmup: int = 20, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请先运行 source scripts/activate_env.sh 并检查 GPU/driver/PyTorch CUDA。")

    if torch.cuda.get_device_capability(0)[0] < 9:
        print("note: fake-FP8 uint8 E4M3 path on this GPU; numbers are not native FP8 Tensor Core performance.")

    for shape in ((1, 4, 128, 64), (1, 8, 256, 64), (1, 4, 256, 128)):
        torch.manual_seed(8)
        q = torch.randn(*shape, device="cuda", dtype=torch.float16)
        k = torch.randn(*shape, device="cuda", dtype=torch.float16)
        v = torch.randn(*shape, device="cuda", dtype=torch.float16)

        expected = flash_attention_fp8_pytorch(q, k, v)
        triton_out = flash_attention_fp8_triton(q, k, v)
        cuda_out = flash_attention_fp8_cuda(q, k, v)
        assert torch.allclose(triton_out, expected, rtol=3e-2, atol=8e-2)
        assert torch.allclose(cuda_out, expected, rtol=3e-2, atol=8e-2)

        pytorch_ms = time_cuda(lambda: flash_attention_fp8_pytorch(q, k, v), warmup=5, repeat=10)
        triton_ms = time_cuda(lambda: flash_attention_fp8_triton(q, k, v))
        cuda_ms = time_cuda(lambda: flash_attention_fp8_cuda(q, k, v))

        print(f"\nshape: {shape}")
        print(f"PyTorch reference: {pytorch_ms:.4f} ms")
        print(f"Triton FP8 FA:     {triton_ms:.4f} ms")
        print(f"CUDA FP8 FA:       {cuda_ms:.4f} ms")
        print(f"Triton max error:  {(triton_out - expected).abs().max().item():.8e}")
        print(f"CUDA max error:    {(cuda_out - expected).abs().max().item():.8e}")


if __name__ == "__main__":
    main()
