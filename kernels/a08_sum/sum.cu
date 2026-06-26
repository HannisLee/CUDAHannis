#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

#define FLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])

__global__ void sum_v1_kernel(const float *x, float *y, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(y, x[idx]);
  }
}

// 每个 block 负责一段，最后 atomicAdd 得到全局 sum。
__global__ void sum_v2_kernel(const float *x, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float input_s[BLOCK_SIZE];
  input_s[tid] = (idx < N) ? x[idx] : 0.0f;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      input_s[tid] += input_s[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(y, input_s[0]);
  }
}

__global__ void sum_v3_kernel(const float *x, float *y, int N) {
  __shared__ float s_y[32];

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  float val = (idx < N) ? x[idx] : 0.0f;

#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  if (lane_id == 0) {
    s_y[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    int warp_num = blockDim.x / WARP_SIZE;
    val = (lane_id < warp_num) ? s_y[lane_id] : 0.0f;

#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (lane_id == 0) {
      atomicAdd(y, val);
    }
  }
}

__global__ void sum_v4_kernel(const float *x, float *y, int N) {
  __shared__ float s_y[32];

  int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  float val = 0.0f;
  if (idx + 3 < N) {
    float4 tmp_x = FLOAT4(x[idx]);
    val = tmp_x.x + tmp_x.y + tmp_x.z + tmp_x.w;
  } else {
    for (int i = 0; i < 4; ++i) {
      if (idx + i < N) {
        val += x[idx + i];
      }
    }
  }

#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  if (lane_id == 0) {
    s_y[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    int warp_num = blockDim.x / WARP_SIZE;
    val = (lane_id < warp_num) ? s_y[lane_id] : 0.0f;

#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (lane_id == 0) {
      atomicAdd(y, val);
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                  \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                  \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                \
    throw std::runtime_error("values must be " #th_type);                     \
  }

void check_sum_args(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  if (!x.is_cuda() || !y.is_cuda()) {
    throw std::runtime_error("x and y must be CUDA tensors");
  }
  if (!x.is_contiguous() || !y.is_contiguous()) {
    throw std::runtime_error("x and y must be contiguous");
  }
  if (y.numel() != 1) {
    throw std::runtime_error("y must have exactly one element");
  }
}

void sum_v1(torch::Tensor x, torch::Tensor y) {
  check_sum_args(x, y);
  int N = x.numel();
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sum_v1_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),
                                 reinterpret_cast<float *>(y.data_ptr()), N);
}

void sum_v2(torch::Tensor x, torch::Tensor y) {
  check_sum_args(x, y);
  int N = x.numel();
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sum_v2_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),
                                 reinterpret_cast<float *>(y.data_ptr()), N);
}

void sum_v3(torch::Tensor x, torch::Tensor y) {
  check_sum_args(x, y);
  int N = x.numel();
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sum_v3_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),
                                 reinterpret_cast<float *>(y.data_ptr()), N);
}

void sum_v4(torch::Tensor x, torch::Tensor y) {
  check_sum_args(x, y);
  int N = x.numel();
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4));
  sum_v4_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),
                                 reinterpret_cast<float *>(y.data_ptr()), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sum_v1)
  TORCH_BINDING_COMMON_EXTENSION(sum_v2)
  TORCH_BINDING_COMMON_EXTENSION(sum_v3)
  TORCH_BINDING_COMMON_EXTENSION(sum_v4)
}
