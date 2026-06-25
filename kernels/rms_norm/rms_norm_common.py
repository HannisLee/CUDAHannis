import torch

# Matches the hardcoded epsilon inside every kernel in rms_norm_f16_f32.cu
# (rsqrtf(variance / K + epsilon)). The CUDA kernels cannot take eps as an
# argument, so we keep it constant across all backends for direct comparison.
EPSILON = 1e-5

# Supported last-dim (K) values per CUDA variant. These mirror the switch
# statements in the DISPATCH_* macros of rms_norm_f16_f32.cu, so they must stay
# in sync with that file. Names are the short variant tags used by the Python
# wrappers; the CUDA binding function for each is in VARIANT_FUNC below.
SUPPORTED_K = {
    "f32": [64, 128, 256, 512, 1024],
    "f32x4": [64, 128, 256, 512, 1024, 2048, 4096],
    "f16_f16": [64, 128, 256, 512, 1024],
    "f16x2_f16": [64, 128, 256, 512, 1024, 2048],
    "f16x8_f16": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "f16x8_f32": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "f16x8_pack_f16": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "f16x8_pack_f32": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "f16_f32": [64, 128, 256, 512, 1024],
}

# CUDA extension binding name (PYBIND11_MODULE in rms_norm_f16_f32.cu) per variant.
VARIANT_FUNC = {
    "f32": "rms_norm_f32",
    "f32x4": "rms_norm_f32x4",
    "f16_f16": "rms_norm_f16_f16",
    "f16x2_f16": "rms_norm_f16x2_f16",
    "f16x8_f16": "rms_norm_f16x8_f16",
    "f16x8_f32": "rms_norm_f16x8_f32",
    "f16x8_pack_f16": "rms_norm_f16x8_pack_f16",
    "f16x8_pack_f32": "rms_norm_f16x8_pack_f32",
    "f16_f32": "rms_norm_f16_f32",
}

# Reduction dtype: "f16" variants accumulate sum-of-squares in half (lower
# precision, larger error), "f32" variants accumulate in float (reference match).
REDUCE_DTYPE = {
    "f32": "f32",
    "f32x4": "f32",
    "f16_f16": "f16",
    "f16x2_f16": "f16",
    "f16x8_f16": "f16",
    "f16x8_f32": "f32",
    "f16x8_pack_f16": "f16",
    "f16x8_pack_f32": "f32",
    "f16_f32": "f32",
}

# Preferred "auto" variant per input dtype: the fastest variant each supports.
AUTO_VARIANT = {
    torch.float32: "f32x4",
    torch.float16: "f16x8_pack_f32",
}

# All variant tags in definition order (used for stable test/bench output).
ALL_VARIANTS = [
    "f32",
    "f32x4",
    "f16_f16",
    "f16x2_f16",
    "f16x8_f16",
    "f16x8_f32",
    "f16x8_pack_f16",
    "f16x8_pack_f32",
    "f16_f32",
]


def variant_dtype(variant: str) -> torch.dtype:
    return torch.float32 if variant.startswith("f32") else torch.float16


def variants_for(dtype: torch.dtype, k: int) -> list[str]:
    """Variant tags whose input dtype matches and whose dispatch supports K."""
    return [v for v in ALL_VARIANTS if variant_dtype(v) == dtype and k in SUPPORTED_K[v]]


def resolve_variant(variant: str, dtype: torch.dtype, k: int) -> str:
    if variant != "auto":
        if variant not in VARIANT_FUNC:
            raise ValueError(
                f"Unknown RMSNorm variant {variant!r}. "
                f"Expected 'auto' or one of {ALL_VARIANTS}."
            )
        if variant_dtype(variant) != dtype:
            raise TypeError(
                f"Variant {variant!r} expects {variant_dtype(variant)}, got {dtype}."
            )
        if k not in SUPPORTED_K[variant]:
            raise ValueError(
                f"Variant {variant!r} does not support K={k}; "
                f"supported K: {SUPPORTED_K[variant]}."
            )
        return variant

    if dtype not in AUTO_VARIANT:
        raise TypeError(f"RMSNorm supports float16/float32, got {dtype}.")
    chosen = AUTO_VARIANT[dtype]
    if k not in SUPPORTED_K[chosen]:
        # Fall back to any variant of this dtype that supports K.
        matches = variants_for(dtype, k)
        if not matches:
            raise ValueError(
                f"No CUDA RMSNorm variant supports dtype={dtype}, K={k}. "
                f"Supported K per variant: {SUPPORTED_K}."
            )
        chosen = matches[-1]
    return chosen


def validate_input(x: torch.Tensor, g: float) -> None:
    if not x.is_cuda:
        raise RuntimeError("RMSNorm expects a CUDA tensor.")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"RMSNorm supports float16/float32, got {x.dtype}.")
    if x.dim() != 2:
        raise ValueError(
            "RMSNorm expects a 2-D (N, K) tensor; the CUDA kernels read "
            "x.size(0)/x.size(1)."
        )
    if not isinstance(g, (int, float)):
        raise TypeError(f"RMSNorm scale g must be a float, got {type(g).__name__}.")
