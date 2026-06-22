from .activation_24_cuda import activation_24_sparsity_cuda
from .activation_24_pytorch import activation_24_sparsity_pytorch
from .activation_24_triton import activation_24_sparsity_triton


def sparsify_before_up_gate(hidden_states, backend: str = "triton"):
    if backend == "pytorch":
        return activation_24_sparsity_pytorch(hidden_states)
    if backend == "triton":
        return activation_24_sparsity_triton(hidden_states)
    if backend == "cuda":
        return activation_24_sparsity_cuda(hidden_states)
    raise ValueError(f"Unknown backend {backend!r}. Expected 'pytorch', 'triton', or 'cuda'.")


__all__ = [
    "activation_24_sparsity_cuda",
    "activation_24_sparsity_pytorch",
    "activation_24_sparsity_triton",
    "sparsify_before_up_gate",
]
