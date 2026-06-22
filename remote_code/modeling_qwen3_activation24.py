import os
import statistics
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3MLP

from kernels.activation_24 import activation_24_sparsity_pytorch, activation_24_sparsity_triton


def _activation24_backend() -> str:
    backend = os.environ.get("QWEN3_ACTIVATION24_BACKEND", "triton").strip().lower()
    if backend not in {"pytorch", "triton"}:
        raise ValueError("QWEN3_ACTIVATION24_BACKEND must be 'pytorch' or 'triton'.")
    return backend


def _sparsify(x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == "pytorch":
        return activation_24_sparsity_pytorch(x)
    if backend == "triton":
        return activation_24_sparsity_triton(x)
    raise ValueError(f"Unknown activation24 backend: {backend!r}")


def _empty_site_stats() -> dict[str, dict[str, Any]]:
    return {"mlp_input": {}, "down_input": {}}


class Qwen3Activation24MLP(Qwen3MLP):
    def __init__(self, config, layer_idx: int, backend: str):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.activation24_backend = backend
        self.activation24_shape_stats = _empty_site_stats()

    def reset_activation24_stats(self) -> None:
        self.activation24_shape_stats = _empty_site_stats()

    def _record_shape(self, site: str, x: torch.Tensor) -> None:
        key = f"{tuple(x.shape)}|{str(x.dtype)}|{str(x.device)}"
        stats = self.activation24_shape_stats[site].setdefault(
            key,
            {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "calls": 0,
                "layers": defaultdict(int),
            },
        )
        stats["calls"] += 1
        stats["layers"][str(self.layer_idx)] += 1

    def _sparse(self, site: str, x: torch.Tensor) -> torch.Tensor:
        self._record_shape(site, x)
        return _sparsify(x, self.activation24_backend)

    def forward(self, x):
        x = self._sparse("mlp_input", x)
        down_input = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_input = self._sparse("down_input", down_input)
        return self.down_proj(down_input)


class Qwen3Activation24ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.activation24_backend = _activation24_backend()
        for layer_idx, layer in enumerate(self.model.layers):
            layer.mlp = Qwen3Activation24MLP(config, layer_idx, self.activation24_backend)

    def reset_activation24_stats(self) -> None:
        for layer in self.model.layers:
            layer.mlp.reset_activation24_stats()

    def get_activation24_stats(self) -> dict[str, dict[str, Any]]:
        merged = _empty_site_stats()
        for layer in self.model.layers:
            for site, site_stats in layer.mlp.activation24_shape_stats.items():
                for key, stats in site_stats.items():
                    dst = merged[site].setdefault(
                        key,
                        {
                            "shape": stats["shape"],
                            "dtype": stats["dtype"],
                            "device": stats["device"],
                            "calls": 0,
                            "layers": {},
                        },
                    )
                    dst["calls"] += stats["calls"]
                    for layer_idx, calls in stats["layers"].items():
                        dst["layers"][layer_idx] = dst["layers"].get(layer_idx, 0) + calls
        return merged

    def benchmark_activation24_shapes(
        self,
        warmup: int = 5,
        repeat: int = 20,
        max_shapes: int = 0,
    ) -> dict[str, list[dict[str, Any]]]:
        stats = self.get_activation24_stats()
        device = next(self.parameters()).device
        if device.type != "cuda":
            return {"error": "activation24 benchmarks require a CUDA model device."}

        benchmarks: dict[str, list[dict[str, Any]]] = {"mlp_input": [], "down_input": []}
        for site, site_stats in stats.items():
            items = sorted(site_stats.values(), key=lambda item: item["calls"], reverse=True)
            if max_shapes > 0:
                items = items[:max_shapes]
            for item in items:
                dtype = _dtype_from_string(item["dtype"])
                x = torch.randn(item["shape"], device=device, dtype=dtype)
                fn = lambda: _sparsify(x, self.activation24_backend)
                timings = _time_cuda(fn, warmup=warmup, repeat=repeat)
                benchmarks[site].append(
                    {
                        "shape": item["shape"],
                        "dtype": item["dtype"],
                        "calls": item["calls"],
                        "warmup": warmup,
                        "repeat": repeat,
                        **timings,
                    }
                )
        return benchmarks


def _dtype_from_string(dtype: str) -> torch.dtype:
    mapping = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported benchmark dtype for activation24: {dtype}")
    return mapping[dtype]


def _time_cuda(fn, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    return {
        "mean_ms": float(statistics.fmean(elapsed_ms)),
        "median_ms": float(statistics.median(elapsed_ms)),
        "min_ms": float(min(elapsed_ms)),
        "max_ms": float(max(elapsed_ms)),
    }


__all__ = ["Qwen3Activation24ForCausalLM", "Qwen3Activation24MLP"]
