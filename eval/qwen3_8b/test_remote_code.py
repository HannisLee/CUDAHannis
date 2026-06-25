import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from kernels.activation_24 import activation_24_sparsity_pytorch, activation_24_sparsity_triton
from eval.qwen3_8b.remote_code.modeling_qwen3_activation24 import Qwen3Activation24ForCausalLM, Qwen3Activation24MLP


def tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for activation24 kernels.")
@pytest.mark.parametrize("backend", ["pytorch", "triton"])
def test_qwen3_activation24_mlp_matches_manual_formula(backend: str) -> None:
    torch.manual_seed(0)
    config = tiny_qwen3_config()
    mlp = Qwen3Activation24MLP(config, layer_idx=0, backend=backend).cuda().half().eval()
    x = torch.randn(2, 3, config.hidden_size, device="cuda", dtype=torch.float16)

    sparse = activation_24_sparsity_pytorch if backend == "pytorch" else activation_24_sparsity_triton
    expected_x = sparse(x)
    expected_down_input = sparse(mlp.act_fn(mlp.gate_proj(expected_x)) * mlp.up_proj(expected_x))
    expected = mlp.down_proj(expected_down_input)

    actual = mlp(x)

    assert torch.allclose(actual, expected, rtol=0, atol=0)
    stats = mlp.activation24_shape_stats
    assert sum(item["calls"] for item in stats["mlp_input"].values()) == 1
    assert sum(item["calls"] for item in stats["down_input"].values()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for activation24 kernels.")
def test_qwen3_activation24_model_collects_stats_and_benchmarks(monkeypatch) -> None:
    monkeypatch.setenv("QWEN3_ACTIVATION24_BACKEND", "triton")
    model = Qwen3Activation24ForCausalLM(tiny_qwen3_config()).cuda().half().eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4), device="cuda")

    with torch.no_grad():
        model(input_ids=input_ids)

    stats = model.get_activation24_stats()
    assert stats["mlp_input"]
    assert stats["down_input"]

    benchmarks = model.benchmark_activation24_shapes(warmup=1, repeat=1, max_shapes=1)
    assert benchmarks["mlp_input"][0]["mean_ms"] >= 0
    assert benchmarks["down_input"][0]["mean_ms"] >= 0
