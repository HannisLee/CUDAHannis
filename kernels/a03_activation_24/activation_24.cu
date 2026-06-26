#include <torch/extension.h>
#include <torch/types.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>


// ============================================================================
// Tensor check macros
// ============================================================================

#define CHECK_CUDA(T)                                      \
  TORCH_CHECK((T).is_cuda(), #T " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(T)                                \
  TORCH_CHECK((T).is_contiguous(), #T " must be contiguous")

#define CHECK_INPUT(T)                                     \
  CHECK_CUDA(T);                                           \
  CHECK_CONTIGUOUS(T)

#define CHECK_ACTIVATION_24_DTYPE(T)                                      \
  TORCH_CHECK(                                                           \
      (T).scalar_type() == torch::kFloat16,                               \
      #T " must be float16"                                               \
  )


// ============================================================================
// Device helper: abs as float
// ============================================================================
//
// 2:4 sparsity 判断大小时，通常比较绝对值。
// 比较 fp16 绝对值时，先转成 float 再取 abs。
// ============================================================================

__device__ __forceinline__ float abs_as_float(c10::Half value) {
  float v = static_cast<float>(value);
  return v < 0.0f ? -v : v;
}


// ============================================================================
// Device helper: rank comparison
// ============================================================================
//
// 当前 lane 是否应该被 other lane 超过。
//
// 规则：
//   1. 绝对值更大的 lane 排名前面；
//   2. 如果绝对值相同，lane index 更小的排前面。
//      这样可以保证 tie-breaking 是确定性的。
// ============================================================================

__device__ __forceinline__ bool should_other_rank_before_current(
    float current_abs,
    float other_abs,
    int current_lane,
    int other_lane
) {
  return other_abs > current_abs ||
         (other_abs == current_abs && other_lane < current_lane);
}


// ============================================================================
// Kernel: activation 2:4 sparsity
// ============================================================================
//
// 功能：
//   对输入 x 的最后一维按连续 4 个元素分组。
//   每组保留绝对值最大的 2 个元素，其余 2 个置零。
//
// 例子：
//   group = [x0, x1, x2, x3]
//   abs 最大的两个保留，其余置 0。
//
// 支持：
//   - float16
//
// grid / block 映射：
//   一个 thread 处理一个 4-element group。
//   group_idx 是 flatten 后的 group 编号。
// ============================================================================

__global__ void activation_24_sparsity_kernel(
    const c10::Half* __restrict__ x,
    c10::Half* __restrict__ out,
    int64_t rows,
    int64_t last_dim,
    int64_t groups_per_row
) {
  const int64_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_groups = rows * groups_per_row;

  if (group_idx >= total_groups) {
    return;
  }

  // 当前 group 属于第几行
  const int64_t row = group_idx / groups_per_row;

  // 当前 group 是这一行里的第几个 4-element group
  const int64_t group = group_idx - row * groups_per_row;

  // 当前 group 在原始 tensor flatten 后的起始位置
  const int64_t base = row * last_dim + group * 4;

  c10::Half values[4];
  float abs_values[4];
  bool valid[4];

#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    const int64_t offset_in_row = group * 4 + lane;

    valid[lane] = offset_in_row < last_dim;

    if (valid[lane]) {
      values[lane] = x[base + lane];
      abs_values[lane] = abs_as_float(values[lane]);
    } else {
      values[lane] = c10::Half(0);
      abs_values[lane] = -3.4028234663852886e38F;
    }
  }

#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    int rank = 0;

    #pragma unroll
        for (int other = 0; other < 4; ++other) {
          if (other != lane &&
              should_other_rank_before_current(
                  abs_values[lane],
                  abs_values[other],
                  lane,
                  other
              )) {
            rank += 1;
          }
        }

    // rank < 2 表示当前 lane 是 top-2，需要保留。
    if (valid[lane]) {
      out[base + lane] = rank < 2 ? values[lane] : c10::Half(0);
    }
  }
}


// ============================================================================
// Activation 2:4 launch macros
// ============================================================================
//
// 这里把 activation_24_sparsity_forward 的 CUDA launch 改成宏风格：
//
//   LAUNCH_ACTIVATION_24_SPARSITY_KERNEL
//     负责真正发射 CUDA kernel。
//
//   DISPATCH_ACTIVATION_24_SPARSITY_KERNEL
//     负责检查 dtype 并完成 kernel launch。
//
// 注意：
//   activation_24_sparsity_kernel 只支持 fp16。
//
//   Python 侧仍然只暴露一个函数：
//     activation_24_sparsity_forward(x)
// ============================================================================

#define ACTIVATION_24_THREADS 256


// ----------------------------------------------------------------------------
// Launch macro
// ----------------------------------------------------------------------------
//
// 这个宏默认使用当前作用域里的变量：
//   x
//   out
//   rows
//   last_dim
//   groups_per_row
//   grid
//   block
//   stream
// ----------------------------------------------------------------------------

#define LAUNCH_ACTIVATION_24_SPARSITY_KERNEL()                             \
  activation_24_sparsity_kernel<<<grid, block, 0, stream>>>(               \
      x.data_ptr<c10::Half>(),                                             \
      out.data_ptr<c10::Half>(),                                           \
      rows,                                                                \
      last_dim,                                                            \
      groups_per_row                                                       \
  );


// ----------------------------------------------------------------------------
// Dispatch macro
// ----------------------------------------------------------------------------
//
// 只支持 torch.float16 -> c10::Half。
// ----------------------------------------------------------------------------

#define DISPATCH_ACTIVATION_24_SPARSITY_KERNEL()                           \
  do {                                                                     \
    dim3 block(ACTIVATION_24_THREADS);                                     \
                                                                           \
    const int64_t blocks64 =                                               \
        (total_groups + ACTIVATION_24_THREADS - 1) /                       \
        ACTIVATION_24_THREADS;                                             \
                                                                           \
    TORCH_CHECK(blocks64 <= 2147483647LL,                                  \
                "activation_24_sparsity: too many CUDA blocks");           \
                                                                           \
    dim3 grid(static_cast<unsigned int>(blocks64));                        \
                                                                           \
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
                                                                           \
    if (x.scalar_type() == torch::kFloat16) {                              \
      LAUNCH_ACTIVATION_24_SPARSITY_KERNEL()                               \
    } else {                                                               \
      TORCH_CHECK(false,                                                   \
                  "activation 2:4 sparsity supports float16");             \
    }                                                                      \
                                                                           \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
  } while (0)


// ============================================================================
// Forward wrapper
// ============================================================================
//
// Python 调用形式：
//   out = lib.activation_24_sparsity_forward(x)
//
// 功能：
//   对输入 x 的最后一维做 activation 2:4 sparsity。
//   每连续 4 个元素为一组，保留绝对值最大的 2 个，其余置 0。
// ============================================================================

torch::Tensor activation_24_sparsity_forward(torch::Tensor x) {
  CHECK_INPUT(x);

  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");

  TORCH_CHECK(
      x.scalar_type() == torch::kFloat16,
      "activation 2:4 sparsity supports float16"
  );

  const c10::cuda::CUDAGuard device_guard(x.device());

  auto out = torch::empty_like(x);

  const int64_t last_dim = x.size(-1);
  const int64_t numel = x.numel();

  if (last_dim == 0 || numel == 0) {
    return out;
  }

  const int64_t rows = numel / last_dim;
  const int64_t groups_per_row = (last_dim + 3) / 4;
  const int64_t total_groups = rows * groups_per_row;

  DISPATCH_ACTIVATION_24_SPARSITY_KERNEL();

  return out;
}


// ============================================================================
// PyTorch binding helpers
// ============================================================================

#define STRINGIFY_IMPL(str) #str
#define STRINGIFY(str) STRINGIFY_IMPL(str)

#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGIFY(func), &func, STRINGIFY(func));


// ============================================================================
// PyTorch binding registration
// ============================================================================
//
// Python 侧可见函数：
//   activation_24_sparsity_forward(x)
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(activation_24_sparsity_forward)
}
