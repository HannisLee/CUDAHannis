#pragma once

#include <c10/util/Half.h>

// Shared device helpers for the activation 2:4 sparsity CUDA kernels.
//
// `keep_lane` is the single source of truth for the lower-index tie-break rule
// shared by every version (and by the PyTorch / Triton references): within a
// 4-element group, a lane keeps its value iff fewer than 2 of the other lanes
// outrank it, where "outranks" means strictly larger magnitude, or equal
// magnitude with a lower lane index.

template <typename scalar_t>
static __device__ __forceinline__ float abs_as_float(scalar_t value) {
  float as_float = static_cast<float>(value);
  return as_float < 0.0f ? -as_float : as_float;
}

// Returns true when `other` outranks `current`: strictly larger magnitude, or
// equal magnitude with a lower lane index.
static __device__ __forceinline__ bool keep_lane(float current, float other,
                                                 int current_lane, int other_lane) {
  return other > current || (other == current && other_lane < current_lane);
}

// A 4-lane group as a single aligned object so a whole group can be loaded and
// stored in one transaction (16 bytes for float32, 8 bytes for float16). The
// base pointer is always properly aligned because each group starts at a
// 16-byte (float32) / 8-byte (float16) boundary.
template <typename scalar_t>
struct alignas(sizeof(scalar_t) * 4) GroupVec {
  scalar_t d[4];
};
