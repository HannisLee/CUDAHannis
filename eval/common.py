from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

import torch


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run `source scripts/activate_env.sh` and check the GPU, "
            "driver, and PyTorch CUDA runtime."
        )


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.lower().replace(",", "x").split("x") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}; use dimensions such as 1x128x4096") from exc
    if not shape or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}; all dimensions must be positive")
    return shape


def selected_shapes(values: Sequence[tuple[int, ...]] | None, defaults: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
    return list(values) if values else list(defaults)


def selected_dtypes(values: Sequence[str] | None, defaults: Sequence[str] = ("float16", "float32")) -> list[torch.dtype]:
    names = list(values) if values else list(defaults)
    return [DTYPES[name] for name in names]


def add_common_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    dtypes: Sequence[str] = ("float16", "float32"),
) -> None:
    parser.add_argument("--shape", action="append", type=parse_shape, help="Repeatable shape such as 1x128x4096.")
    parser.add_argument("--dtype", action="append", choices=dtypes, help="Repeat to benchmark multiple dtypes.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)


def validate_benchmark_counts(warmup: int, repeat: int) -> None:
    if warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if repeat <= 0:
        raise ValueError("--repeat must be positive")


def time_cuda(fn: Callable[[], Any], *, warmup: int, repeat: int) -> float:
    validate_benchmark_counts(warmup, repeat)
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
    return float(start.elapsed_time(end) / repeat)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


def print_backend_result(
    backend: str,
    latency_ms: float,
    *,
    reference_ms: float,
    metrics: dict[str, float] | None = None,
) -> None:
    speedup = reference_ms / latency_ms if latency_ms > 0 else float("inf")
    fields = [f"backend={backend}", f"latency_ms={latency_ms:.6f}", f"speedup_vs_pytorch={speedup:.3f}"]
    if metrics:
        fields.extend(f"{name}={value:.8e}" for name, value in metrics.items())
    print(" ".join(fields))

