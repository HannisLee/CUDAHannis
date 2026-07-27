import torch

torch.set_grad_enabled(False)


def sgemm_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Naive reference: C = A @ B,  A: M×K, B: K×N  ->  C: M×N"""
    assert a.dim() == 2 and b.dim() == 2
    assert a.size(1) == b.size(0), f"K mismatch: A is {tuple(a.shape)}, B is {tuple(b.shape)}"
    return a @ b


if __name__ == "__main__":
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = sgemm_naive(a, b)
    print("C shape:", tuple(c.shape))
    print("first values:", c.flatten()[:3].tolist())
