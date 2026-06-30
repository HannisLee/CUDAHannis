#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define V5_ITEMS_PER_THREAD 8
#define V6_BLOCK_SIZE 128
#define V6_ITEMS_PER_THREAD 8
#define V6_ELEMENTS_PER_BLOCK (V6_BLOCK_SIZE * V6_ITEMS_PER_THREAD)
#define FLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])

__global__ void max_v0_kernel(const float *x, float *y, int N) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; ++i) {
      max_val = fmaxf(max_val, x[i]);
    }
    y[0] = max_val;
  }
}

__global__ void max_v1_stage_kernel(const float *x, float *partial, int N) {
  int tid = threadIdx.x;
  int idx = blockDim.x * blockIdx.x + tid;

  __shared__ float smem[BLOCK_SIZE];
  smem[tid] = (idx < N) ? x[idx] : -FLT_MAX;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x] = smem[0];
  }
}

__global__ void max_v2_stage_kernel(const float *x, float *partial, int N) {
  __shared__ float warp_max_s[32];

  int tid = threadIdx.x;
  int idx = blockDim.x * blockIdx.x + tid;
  int lane_id = tid & 31;
  int warp_id = tid >> 5;

  float val = (idx < N) ? x[idx] : -FLT_MAX;

#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
    val = fmaxf(val, other);
  }

  if (lane_id == 0) {
    warp_max_s[warp_id] = val;
  }
  __syncthreads();

  int warp_num = blockDim.x / WARP_SIZE;
  if (warp_id == 0) {
    val = (lane_id < warp_num) ? warp_max_s[lane_id] : -FLT_MAX;

#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
      val = fmaxf(val, other);
    }

    if (lane_id == 0) {
      partial[blockIdx.x] = val;
    }
  }
}

__global__ void max_v4_stage_kernel(const float *x, float *partial, int N) {
  __shared__ float warp_max_s[32];

  int tid = threadIdx.x;
  int idx = (blockDim.x * blockIdx.x + tid) * 4;
  int lane_id = tid & 31;
  int warp_id = tid >> 5;

  float val = -FLT_MAX;
  if (idx + 3 < N) {
    float4 reg_x = FLOAT4(x[idx]);
    val = fmaxf(fmaxf(reg_x.x, reg_x.y), fmaxf(reg_x.z, reg_x.w));
  } else {
    for (int i = 0; i < 4; ++i) {
      if (idx + i < N) {
        val = fmaxf(val, x[idx + i]);
      }
    }
  }

#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
    val = fmaxf(val, other);
  }

  if (lane_id == 0) {
    warp_max_s[warp_id] = val;
  }
  __syncthreads();

  int warp_num = blockDim.x / WARP_SIZE;
  if (warp_id == 0) {
    val = (lane_id < warp_num) ? warp_max_s[lane_id] : -FLT_MAX;

#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
      val = fmaxf(val, other);
    }

    if (lane_id == 0) {
      partial[blockIdx.x] = val;
    }
  }
}

__global__ void max_v5_kernel(const float *x, float *partial, float *y,
                              int *counter, int N) {
  __shared__ float warp_max_s[32];
  __shared__ bool is_last_block;

  int tid = threadIdx.x;
  int idx = (blockDim.x * blockIdx.x + tid) * V5_ITEMS_PER_THREAD;
  int lane_id = tid & 31;
  int warp_id = tid >> 5;

  float val = -FLT_MAX;
  #pragma unroll
  for (int i = 0; i < V5_ITEMS_PER_THREAD; i += 4) {
    int load_idx = idx + i;
    if (load_idx + 3 < N) {
      float4 reg_x = FLOAT4(x[load_idx]);
      val = fmaxf(val, fmaxf(fmaxf(reg_x.x, reg_x.y),
                             fmaxf(reg_x.z, reg_x.w)));
    } else {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (load_idx + j < N) {
          val = fmaxf(val, x[load_idx + j]);
        }
      }
    }
  }

#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
    val = fmaxf(val, other);
  }

  if (lane_id == 0) {
    warp_max_s[warp_id] = val;
  }
  __syncthreads();

  int warp_num = blockDim.x / WARP_SIZE;
  if (warp_id == 0) {
    val = (lane_id < warp_num) ? warp_max_s[lane_id] : -FLT_MAX;

#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
      val = fmaxf(val, other);
    }
  }

  if (tid == 0) {
    partial[blockIdx.x] = val;
    __threadfence();
    int ticket = atomicAdd(counter, 1);
    is_last_block = (ticket == gridDim.x - 1);
  }
  __syncthreads();

  if (is_last_block) {
    float final_val = -FLT_MAX;
    for (int i = tid; i < gridDim.x; i += blockDim.x) {
      final_val = fmaxf(final_val, partial[i]);
    }

#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      float other = __shfl_down_sync(0xFFFFFFFF, final_val, offset);
      final_val = fmaxf(final_val, other);
    }

    if (lane_id == 0) {
      warp_max_s[warp_id] = final_val;
    }
    __syncthreads();

    if (warp_id == 0) {
      final_val = (lane_id < warp_num) ? warp_max_s[lane_id] : -FLT_MAX;

#pragma unroll
      for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, final_val, offset);
        final_val = fmaxf(final_val, other);
      }

      if (lane_id == 0) {
        y[0] = final_val;
        *counter = 0;
      }
    }
  }
}

__global__ void max_v6_stage_kernel(const float *x, float *partial, int N) {
  __shared__ float warp_max_s[V6_BLOCK_SIZE / WARP_SIZE];

  int tid = threadIdx.x;
  int lane_id = tid & 31;
  int warp_id = tid >> 5;
  int block_base = blockIdx.x * V6_ELEMENTS_PER_BLOCK;
  int idx0 = block_base + tid * 4;
  int idx1 = block_base + (V6_ELEMENTS_PER_BLOCK / 2) + tid * 4;

  float val = -FLT_MAX;
  if (idx0 + 3 < N) {
    float4 reg_x = FLOAT4(x[idx0]);
    val = fmaxf(fmaxf(reg_x.x, reg_x.y), fmaxf(reg_x.z, reg_x.w));
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (idx0 + i < N) {
        val = fmaxf(val, x[idx0 + i]);
      }
    }
  }

  if (idx1 + 3 < N) {
    float4 reg_x = FLOAT4(x[idx1]);
    val = fmaxf(val, fmaxf(fmaxf(reg_x.x, reg_x.y),
                           fmaxf(reg_x.z, reg_x.w)));
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (idx1 + i < N) {
        val = fmaxf(val, x[idx1 + i]);
      }
    }
  }

#pragma unroll
  for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
    float other = __shfl_xor_sync(0xFFFFFFFF, val, mask);
    val = fmaxf(val, other);
  }

  if (lane_id == 0) {
    warp_max_s[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = (lane_id < (V6_BLOCK_SIZE / WARP_SIZE)) ? warp_max_s[lane_id]
                                                  : -FLT_MAX;

#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
      float other = __shfl_xor_sync(0xFFFFFFFF, val, mask);
      val = fmaxf(val, other);
    }

    if (lane_id == 0) {
      partial[blockIdx.x] = val;
    }
  }
}

__global__ void max_finalize_kernel(const float *partial, float *y,
                                    int n_partials) {
  int tid = threadIdx.x;
  float val = -FLT_MAX;

  for (int i = tid; i < n_partials; i += blockDim.x) {
    val = fmaxf(val, partial[i]);
  }

  __shared__ float smem[BLOCK_SIZE];
  smem[tid] = val;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[0] = smem[0];
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

void check_max_args(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  if (!x.is_cuda() || !y.is_cuda()) {
    throw std::runtime_error("x and y must be CUDA tensors");
  }
  if (!x.is_contiguous() || !y.is_contiguous()) {
    throw std::runtime_error("x and y must be contiguous");
  }
  if (x.numel() <= 0) {
    throw std::runtime_error("x must have at least one element");
  }
  if (y.numel() != 1) {
    throw std::runtime_error("y must have exactly one element");
  }
}

void max_v0(torch::Tensor x, torch::Tensor y) {
  check_max_args(x, y);
  int N = x.numel();
  max_v0_kernel<<<1, 1>>>(reinterpret_cast<float *>(x.data_ptr()),
                          reinterpret_cast<float *>(y.data_ptr()), N);
}

void max_v1(torch::Tensor x, torch::Tensor y) {
  check_max_args(x, y);
  int N = x.numel();
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  auto partial = torch::empty({grid.x}, x.options());

  max_v1_stage_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),
                                       reinterpret_cast<float *>(partial.data_ptr()),
                                       N);
  max_finalize_kernel<<<1, block>>>(
      reinterpret_cast<float *>(partial.data_ptr()),
      reinterpret_cast<float *>(y.data_ptr()), grid.x);
}

void max_v2(torch::Tensor x, torch::Tensor y) {
  check_max_args(x, y);
  int N = x.numel();
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  auto partial = torch::empty({grid.x}, x.options());

  max_v2_stage_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),
                                       reinterpret_cast<float *>(partial.data_ptr()),
                                       N);
  max_finalize_kernel<<<1, block>>>(
      reinterpret_cast<float *>(partial.data_ptr()),
      reinterpret_cast<float *>(y.data_ptr()), grid.x);
}

void max_v4(torch::Tensor x, torch::Tensor y) {
  check_max_args(x, y);
  int N = x.numel();
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4));
  auto partial = torch::empty({grid.x}, x.options());

  max_v4_stage_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),
                                       reinterpret_cast<float *>(partial.data_ptr()),
                                       N);
  max_finalize_kernel<<<1, block>>>(
      reinterpret_cast<float *>(partial.data_ptr()),
      reinterpret_cast<float *>(y.data_ptr()), grid.x);
}

void max_v5(torch::Tensor x, torch::Tensor y) {
  check_max_args(x, y);
  int N = x.numel();
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE * V5_ITEMS_PER_THREAD - 1) /
            (BLOCK_SIZE * V5_ITEMS_PER_THREAD));
  auto partial_and_counter = torch::empty({grid.x + 1}, x.options());

  float *partial_ptr = reinterpret_cast<float *>(partial_and_counter.data_ptr());
  int *counter_ptr = reinterpret_cast<int *>(partial_ptr + grid.x);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(counter_ptr, 0, sizeof(int), stream);

  max_v5_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<float *>(x.data_ptr()), partial_ptr,
      reinterpret_cast<float *>(y.data_ptr()), counter_ptr, N);
}

void max_v6(torch::Tensor x, torch::Tensor y) {
  check_max_args(x, y);
  int current_n = x.numel();
  const float *current_ptr = reinterpret_cast<const float *>(x.data_ptr());
  dim3 block(V6_BLOCK_SIZE);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  std::vector<torch::Tensor> partials;

  if (current_n == 1) {
    cudaMemcpyAsync(reinterpret_cast<float *>(y.data_ptr()), current_ptr,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);
    return;
  }

  while (current_n > 1) {
    int n_blocks = (current_n + V6_ELEMENTS_PER_BLOCK - 1) /
                   V6_ELEMENTS_PER_BLOCK;
    float *out_ptr = nullptr;
    if (n_blocks == 1) {
      out_ptr = reinterpret_cast<float *>(y.data_ptr());
    } else {
      partials.push_back(torch::empty({n_blocks}, x.options()));
      out_ptr = reinterpret_cast<float *>(partials.back().data_ptr());
    }

    max_v6_stage_kernel<<<n_blocks, block, 0, stream>>>(current_ptr, out_ptr,
                                                        current_n);
    current_ptr = out_ptr;
    current_n = n_blocks;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(max_v0)
  TORCH_BINDING_COMMON_EXTENSION(max_v1)
  TORCH_BINDING_COMMON_EXTENSION(max_v2)
  TORCH_BINDING_COMMON_EXTENSION(max_v4)
  TORCH_BINDING_COMMON_EXTENSION(max_v5)
  TORCH_BINDING_COMMON_EXTENSION(max_v6)
}
