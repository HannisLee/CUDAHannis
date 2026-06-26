#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])


// ============================================================================
// Tensor check macros
// ============================================================================
//
// 这些宏用于在 C++/CUDA launcher 入口处检查 Python 传进来的 Tensor：
//   - 是否在 CUDA 上
//   - 是否连续存储
//   - dtype 是否为 torch.float16
//   - shape 是否一致
//   - device 是否一致
//
// 注意：
//   kernel 内部不应该做这些检查，检查应该放在 launcher 层。
//   因为 kernel 运行在 GPU 上，错误处理和异常不方便。
// ============================================================================

#define CHECK_CUDA(T)                                      \
  TORCH_CHECK((T).is_cuda(), #T " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(T)                                \
  TORCH_CHECK((T).is_contiguous(), #T " must be contiguous")

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)               \
  do {                                                     \
    if ((T).scalar_type() != (th_type)) {                  \
      std::cout << "Tensor Info: " << (T).options()        \
                << std::endl;                             \
      throw std::runtime_error("Tensor must be " #th_type);\
    }                                                      \
  } while (0)

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                   \
  do {                                                     \
    TORCH_CHECK((T1).dim() == (T2).dim(),                  \
                #T1 " and " #T2 " dim mismatch");         \
    for (int64_t i = 0; i < (T1).dim(); ++i) {             \
      TORCH_CHECK((T1).size(i) == (T2).size(i),            \
                  #T1 " and " #T2 " shape mismatch");     \
    }                                                      \
  } while (0)

#define CHECK_TORCH_TENSOR_DEVICE(T1, T2)                  \
  TORCH_CHECK((T1).device() == (T2).device(),              \
              #T1 " and " #T2 " must be on same device")

#define CHECK_INPUT(T)                                     \
  CHECK_CUDA(T);                                           \
  CHECK_CONTIGUOUS(T)


// ============================================================================
// Kernel 1: scalar FP16 vector add
// ============================================================================
//
// 功能：
//   c[i] = a[i] + b[i]
//
// 特点：
//   - 每个 CUDA thread 处理 1 个 half 元素；
//   - 使用 __hadd 做 half 加法；
//   - 最简单，适合作为 baseline；
//   - memory access 是连续的，具备 coalesced memory access 条件。
//
// grid / block 映射：
//   idx = blockIdx.x * blockDim.x + threadIdx.x
//
// 对于 2D tensor 的特殊发射方式：
//   如果 a shape 是 [S, K]，并且 blockDim.x = K，gridDim.x = S，
//   那么 idx = row * K + col，刚好对应一行。
// ============================================================================

__global__ void vector_add_f16_kernel(
    const __half* a,
    const __half* b,
    __half* c,
    int64_t n
) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    c[idx] = __hadd(a[idx], b[idx]);
  }
}


// ============================================================================
// Kernel 2: FP16x2 vector add
// ============================================================================
//
// 功能：
//   每个 thread 处理 2 个 half，也就是 1 个 half2。
//
// 为什么用 half2：
//   half2 是 CUDA 提供的向量化 half 类型，一个 half2 包含两个 half。
//   __hadd2 可以一次完成两个 half 的加法：
//     c2 = a2 + b2
//
// 优点：
//   - 每个 thread 处理更多元素；
//   - 减少 thread 数量；
//   - 对连续 FP16 elementwise 操作通常更高效。
//
// 尾部处理：
//   如果 n 是奇数，最后一个 half 无法组成完整 half2，
//   所以用 scalar __hadd 单独处理。
// ============================================================================

__global__ void vector_add_f16x2_kernel(
    const __half* a,
    const __half* b,
    __half* c,
    int64_t n
) {
  const int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t base = pack_idx * 2;

  if (base + 1 < n) {
    const __half2 a2 = *reinterpret_cast<const __half2*>(a + base);
    const __half2 b2 = *reinterpret_cast<const __half2*>(b + base);

    const __half2 c2 = __hadd2(a2, b2);

    *reinterpret_cast<__half2*>(c + base) = c2;
  } else if (base < n) {
    c[base] = __hadd(a[base], b[base]);
  }
}


// ============================================================================
// Kernel 3: FP16x8 vector add
// ============================================================================
//
// 功能：
//   每个 thread 处理 8 个 half。
//   内部拆成 4 次 half2 操作。
//
// 对应关系：
//   8 个 half = 4 个 half2
//
// 优点：
//   - 一个 thread 处理更多连续元素；
//   - 减少 thread 数量；
//   - loop unroll 后指令更规整。
//
// 注意：
//   这里虽然每个 thread 处理 8 个 half，
//   但 load/store 仍然是按 half2 分 4 次完成，
//   不是一次性 128-bit load/store。
//   真正 128-bit 整包搬运在 vector_add_f16x8_pack_kernel 里。
// ============================================================================

__global__ void vector_add_f16x8_kernel(
    const __half* a,
    const __half* b,
    __half* c,
    int64_t n
) {
  const int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t base = pack_idx * 8;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t offset = base + i * 2;

    if (offset + 1 < n) {
      const __half2 a2 = *reinterpret_cast<const __half2*>(a + offset);
      const __half2 b2 = *reinterpret_cast<const __half2*>(b + offset);

      const __half2 c2 = __hadd2(a2, b2);

      *reinterpret_cast<__half2*>(c + offset) = c2;
    } else if (offset < n) {
      c[offset] = __hadd(a[offset], b[offset]);
    }
  }
}


// ============================================================================
// Kernel 4: FP16x8 pack vector add
// ============================================================================
//
// 功能：
//   每个 thread 处理 8 个 half。
//   对完整的 8-half pack，使用 128-bit load/store。
//   对尾部不足 8 个 half 的部分，退回 scalar 处理。
//
// 数据搬运：
//   - load a[base : base + 8] as float4
//   - load b[base : base + 8] as float4
//   - 在寄存器/local pack 里逐元素做 half add
//   - store c[base : base + 8] as float4
//
// 为什么加 alignas(16)：
//   因为 pack_a / pack_b / pack_c 会被 reinterpret_cast 成 float4*。
//   float4 需要 16-byte 对齐。
// ============================================================================

__global__ void vector_add_f16x8_pack_kernel(
    const __half* a,
    const __half* b,
    __half* c,
    int64_t n
) {
  const int64_t pack_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t base = pack_idx * 8;

  if (base + 7 < n) {
    alignas(16) __half pack_a[8];
    alignas(16) __half pack_b[8];
    alignas(16) __half pack_c[8];

    LDST128BITS(pack_a[0]) = *reinterpret_cast<const float4*>(a + base);
    LDST128BITS(pack_b[0]) = *reinterpret_cast<const float4*>(b + base);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      pack_c[i] = __hadd(pack_a[i], pack_b[i]);
    }

    *reinterpret_cast<float4*>(c + base) = LDST128BITS(pack_c[0]);
    return;
  }

  // 处理最后不足 8 个 half 的尾部元素。
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int64_t offset = base + i;
    if (offset < n) {
      c[offset] = __hadd(a[offset], b[offset]);
    }
  }
}


// ============================================================================
// Common input check
// ============================================================================
//
// 三个输入 tensor：
//   a: input tensor
//   b: input tensor
//   c: output tensor
//
// 要求：
//   - 都必须是 CUDA tensor；
//   - 都必须 contiguous；
//   - dtype 都必须是 torch.float16；
//   - shape 必须完全一致；
//   - device 必须一致。
// ============================================================================

void check_vector_add_f16_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c
) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);

  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat16);
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat16);
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat16);

  CHECK_TORCH_TENSOR_SHAPE(a, b);
  CHECK_TORCH_TENSOR_SHAPE(a, c);

  CHECK_TORCH_TENSOR_DEVICE(a, b);
  CHECK_TORCH_TENSOR_DEVICE(a, c);
}


// ============================================================================
// VectorAdd launch macros
// ============================================================================
//
// 这里把 vector_add 的 launcher 改成类似 RMSNorm 的宏调用风格：
//
//   LAUNCH_VECTOR_ADD_XXX_KERNEL
//     负责真正发射 CUDA kernel。
//
//   DISPATCH_VECTOR_ADD_XXX_KERNEL
//     负责计算 grid / block / stream，然后调用 LAUNCH 宏。
//
// 说明：
//   vector_add 是普通 elementwise 算子，不需要像 RMSNorm 那样根据 K
//   switch 到不同 template 版本，所以这里不需要 switch-case。
// ============================================================================

#define VECTOR_ADD_ELEMENTS_PER_BLOCK 256


// ----------------------------------------------------------------------------
// common dispatch helper
// ----------------------------------------------------------------------------
//
// 参数：
//   KERNEL_NAME           : 要发射的 kernel 名字
//   ELEMENTS_PER_THREAD   : 每个 thread 处理几个 half
//   N                     : 总元素数量
//
// 逻辑：
//   1. 如果 a 是 2D tensor [S, K]，并且 K 能被 ELEMENTS_PER_THREAD 整除，
//      优先使用一行一个 block。
//   2. 否则使用普通 1D flatten launch。
// ----------------------------------------------------------------------------

#define DISPATCH_VECTOR_ADD_KERNEL(KERNEL_NAME, ELEMENTS_PER_THREAD, N)        \
  do {                                                                        \
    dim3 grid;                                                                \
    dim3 block;                                                               \
                                                                              \
    if (a.dim() == 2) {                                                       \
      const int64_t s = a.size(0);                                            \
      const int64_t k = a.size(1);                                            \
      const int64_t threads = k / (ELEMENTS_PER_THREAD);                      \
                                                                              \
      if (k % (ELEMENTS_PER_THREAD) == 0 && threads > 0 && threads <= 1024) { \
        grid = dim3(static_cast<unsigned int>(s));                            \
        block = dim3(static_cast<unsigned int>(threads));                     \
      } else {                                                                \
        const int64_t threads_1d =                                            \
            VECTOR_ADD_ELEMENTS_PER_BLOCK / (ELEMENTS_PER_THREAD);            \
        const int64_t grid_1d =                                               \
            ((N) + VECTOR_ADD_ELEMENTS_PER_BLOCK - 1) /                       \
            VECTOR_ADD_ELEMENTS_PER_BLOCK;                                    \
        grid = dim3(static_cast<unsigned int>(grid_1d));                      \
        block = dim3(static_cast<unsigned int>(threads_1d));                  \
      }                                                                       \
    } else {                                                                  \
      const int64_t threads_1d =                                              \
          VECTOR_ADD_ELEMENTS_PER_BLOCK / (ELEMENTS_PER_THREAD);              \
      const int64_t grid_1d =                                                 \
          ((N) + VECTOR_ADD_ELEMENTS_PER_BLOCK - 1) /                         \
          VECTOR_ADD_ELEMENTS_PER_BLOCK;                                      \
      grid = dim3(static_cast<unsigned int>(grid_1d));                        \
      block = dim3(static_cast<unsigned int>(threads_1d));                    \
    }                                                                         \
                                                                              \
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();                   \
                                                                              \
    KERNEL_NAME<<<grid, block, 0, stream>>>(                                  \
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),              \
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),              \
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),                    \
        (N)                                                                   \
    );                                                                        \
                                                                              \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                           \
  } while (0)


// ----------------------------------------------------------------------------
// scalar f16
// 每个 thread 处理 1 个 half
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_ADD_F16_KERNEL(N)                                       \
  DISPATCH_VECTOR_ADD_KERNEL(vector_add_f16_kernel, 1, (N))


// ----------------------------------------------------------------------------
// f16x2
// 每个 thread 处理 2 个 half = 1 个 half2
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_ADD_F16X2_KERNEL(N)                                     \
  DISPATCH_VECTOR_ADD_KERNEL(vector_add_f16x2_kernel, 2, (N))


// ----------------------------------------------------------------------------
// f16x8
// 每个 thread 处理 8 个 half = 4 个 half2
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_ADD_F16X8_KERNEL(N)                                     \
  DISPATCH_VECTOR_ADD_KERNEL(vector_add_f16x8_kernel, 8, (N))


// ----------------------------------------------------------------------------
// f16x8_pack
// 每个 thread 处理 8 个 half，完整 pack 使用 128-bit load/store
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_ADD_F16X8_PACK_KERNEL(N)                                \
  DISPATCH_VECTOR_ADD_KERNEL(vector_add_f16x8_pack_kernel, 8, (N))


// ============================================================================
// Launcher 1: scalar f16
// ============================================================================
//
// Python 调用：
//   lib.vector_add_f16(a, b, c)
// ============================================================================

void vector_add_f16(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c
) {
  check_vector_add_f16_inputs(a, b, c);

  const c10::cuda::CUDAGuard device_guard(a.device());

  const int64_t n = a.numel();
  if (n == 0) {
    return;
  }

  LAUNCH_VECTOR_ADD_F16_KERNEL(n);
}


// ============================================================================
// Launcher 2: f16x2
// ============================================================================
//
// Python 调用：
//   lib.vector_add_f16x2(a, b, c)
// ============================================================================

void vector_add_f16x2(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c
) {
  check_vector_add_f16_inputs(a, b, c);

  const c10::cuda::CUDAGuard device_guard(a.device());

  const int64_t n = a.numel();
  if (n == 0) {
    return;
  }

  LAUNCH_VECTOR_ADD_F16X2_KERNEL(n);
}


// ============================================================================
// Launcher 3: f16x8
// ============================================================================
//
// Python 调用：
//   lib.vector_add_f16x8(a, b, c)
// ============================================================================

void vector_add_f16x8(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c
) {
  check_vector_add_f16_inputs(a, b, c);

  const c10::cuda::CUDAGuard device_guard(a.device());

  const int64_t n = a.numel();
  if (n == 0) {
    return;
  }

  LAUNCH_VECTOR_ADD_F16X8_KERNEL(n);
}


// ============================================================================
// Launcher 4: f16x8_pack
// ============================================================================
//
// Python 调用：
//   lib.vector_add_f16x8_pack(a, b, c)
// ============================================================================

void vector_add_f16x8_pack(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c
) {
  check_vector_add_f16_inputs(a, b, c);

  const c10::cuda::CUDAGuard device_guard(a.device());

  const int64_t n = a.numel();
  if (n == 0) {
    return;
  }

  LAUNCH_VECTOR_ADD_F16X8_PACK_KERNEL(n);
}


// ============================================================================
// Return API template helper
// ============================================================================
//
// inplace-output 版本：
//   vector_add_xxx(a, b, c)
//
// return 版本：
//   c = vector_add_xxx_return(a, b)
// ============================================================================

template <typename LauncherFn>
torch::Tensor vector_add_return_impl(
    torch::Tensor a,
    torch::Tensor b,
    LauncherFn launcher
) {
  auto c = torch::empty_like(a);
  launcher(a, b, c);
  return c;
}


// ============================================================================
// Return API wrappers
// ============================================================================

torch::Tensor vector_add_f16_return(torch::Tensor a, torch::Tensor b) {
  return vector_add_return_impl(a, b, vector_add_f16);
}

torch::Tensor vector_add_f16x2_return(torch::Tensor a, torch::Tensor b) {
  return vector_add_return_impl(a, b, vector_add_f16x2);
}

torch::Tensor vector_add_f16x8_return(torch::Tensor a, torch::Tensor b) {
  return vector_add_return_impl(a, b, vector_add_f16x8);
}

torch::Tensor vector_add_f16x8_pack_return(torch::Tensor a, torch::Tensor b) {
  return vector_add_return_impl(a, b, vector_add_f16x8_pack);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16x8_pack)

  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16_return)
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16x2_return)
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16x8_return)
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16x8_pack_return)
}