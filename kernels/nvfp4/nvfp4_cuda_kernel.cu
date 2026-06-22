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
__device__ __forceinline__ float to_float(scalar_t value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ uint8_t fp4_e2m1_magnitude_code(float value) {
  uint8_t code = 0;
  code += value > 0.250001f;
  code += value > 0.750001f;
  code += value > 1.250001f;
  code += value > 1.750001f;
  code += value > 2.500001f;
  code += value > 3.500001f;
  code += value > 5.000001f;
  return code;
}

template <typename scalar_t>
__global__ void nvfp4_quantize_kernel(const scalar_t* __restrict__ x,
                                      uint8_t* __restrict__ packed,
                                      float* __restrict__ scales,
                                      int64_t rows,
                                      int64_t last_dim,
                                      int64_t blocks_per_row) {
  int64_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_blocks = rows * blocks_per_row;
  if (block_idx >= total_blocks) {
    return;
  }

  int64_t row = block_idx / blocks_per_row;
  int64_t block = block_idx - row * blocks_per_row;
  int64_t row_base = row * last_dim;
  int64_t block_offset = block * 16;

  float values[16];
  float max_abs = 0.0f;
#pragma unroll
  for (int lane = 0; lane < 16; ++lane) {
    int64_t offset = block_offset + lane;
    float value = 0.0f;
    if (offset < last_dim) {
      value = to_float(x[row_base + offset]);
      float abs_value = value < 0.0f ? -value : value;
      max_abs = abs_value > max_abs ? abs_value : max_abs;
    }
    values[lane] = value;
  }

  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  scales[block_idx] = scale;

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    uint8_t packed_byte = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int lane = pair * 2 + j;
      int64_t offset = block_offset + lane;
      uint8_t code = 0;
      if (offset < last_dim) {
        float value = values[lane];
        float abs_scaled = value < 0.0f ? -value / scale : value / scale;
        abs_scaled = abs_scaled > 6.0f ? 6.0f : abs_scaled;
        code = fp4_e2m1_magnitude_code(abs_scaled);
        if (value < 0.0f) {
          code |= 0x08;
        }
      }
      packed_byte |= j == 0 ? code : static_cast<uint8_t>(code << 4);
    }
    packed[row * blocks_per_row * 8 + block * 8 + pair] = packed_byte;
  }
}

std::vector<torch::Tensor> nvfp4_quantize_forward(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16, "NVFP4 supports float16/float32");

  const auto last_dim = x.size(-1);
  TORCH_CHECK(last_dim > 0, "last dimension must be non-empty");
  const auto padded_last_dim = ((last_dim + 15) / 16) * 16;
  const auto rows = x.numel() / last_dim;
  const auto blocks_per_row = padded_last_dim / 16;

  auto packed_shape = x.sizes().vec();
  packed_shape.back() = padded_last_dim / 2;
  auto scale_shape = x.sizes().vec();
  scale_shape.back() = blocks_per_row;

  auto packed = torch::empty(packed_shape, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, x.options().dtype(torch::kFloat32));

  constexpr int threads = 256;
  const int blocks = static_cast<int>((rows * blocks_per_row + threads - 1) / threads);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kFloat32) {
    nvfp4_quantize_kernel<float><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), packed.data_ptr<uint8_t>(), scales.data_ptr<float>(), rows, last_dim, blocks_per_row);
  } else {
    nvfp4_quantize_kernel<c10::Half><<<blocks, threads, 0, stream>>>(
        x.data_ptr<c10::Half>(), packed.data_ptr<uint8_t>(), scales.data_ptr<float>(), rows, last_dim, blocks_per_row);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}
