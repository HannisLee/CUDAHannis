#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Half.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__device__ __forceinline__ float abs_as_float(scalar_t value) {
  float as_float = static_cast<float>(value);
  return as_float < 0.0f ? -as_float : as_float;
}

__device__ __forceinline__ bool keep_lane(float current, float other, int current_lane, int other_lane) {
  return other > current || (other == current && other_lane < current_lane);
}

template <typename scalar_t>
__global__ void activation_24_sparsity_kernel(const scalar_t* __restrict__ x,
                                              scalar_t* __restrict__ out,
                                              int64_t rows,
                                              int64_t last_dim,
                                              int64_t groups_per_row) {
  int64_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_groups = rows * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  int64_t row = group_idx / groups_per_row;
  int64_t group = group_idx - row * groups_per_row;
  int64_t base = row * last_dim + group * 4;

  scalar_t values[4];
  float abs_values[4];
  bool valid[4];

#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    int64_t offset = group * 4 + lane;
    valid[lane] = offset < last_dim;
    if (valid[lane]) {
      values[lane] = x[base + lane];
      abs_values[lane] = abs_as_float(values[lane]);
    } else {
      values[lane] = scalar_t(0);
      abs_values[lane] = -3.4028234663852886e38F;
    }
  }

#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    int rank = 0;
#pragma unroll
    for (int other = 0; other < 4; ++other) {
      if (other != lane && keep_lane(abs_values[lane], abs_values[other], lane, other)) {
        rank += 1;
      }
    }
    if (valid[lane]) {
      out[base + lane] = rank < 2 ? values[lane] : scalar_t(0);
    }
  }
}

torch::Tensor activation_24_sparsity_forward(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
      "activation 2:4 sparsity supports float16/float32");

  auto out = torch::empty_like(x);
  const auto last_dim = x.size(-1);
  if (last_dim == 0 || x.numel() == 0) {
    return out;
  }

  const auto rows = x.numel() / last_dim;
  const auto groups_per_row = (last_dim + 3) / 4;
  const auto total_groups = rows * groups_per_row;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total_groups + threads - 1) / threads);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kFloat32) {
    activation_24_sparsity_kernel<float><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), rows, last_dim, groups_per_row);
  } else {
    activation_24_sparsity_kernel<c10::Half><<<blocks, threads, 0, stream>>>(
        x.data_ptr<c10::Half>(), out.data_ptr<c10::Half>(), rows, last_dim, groups_per_row);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
