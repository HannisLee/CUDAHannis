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
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


// ============================================================================
// GELU (tanh 近似) 核心公式
// ============================================================================
//
//   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ) )
//
// 这里统一在 fp32 下计算，再按需转回 half/fp32。
// 对应 PyTorch 的 F.gelu(x, approximate='tanh')。
// ============================================================================

__device__ __forceinline__ float gelu_tanh_fp32(float x) {
  const float kAlpha = 0.7978845608028654f;  // sqrt(2 / pi)
  const float kBeta  = 0.044715f;
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}


// ----------------------------------------------------------------------------
// fp32 scalar kernel
// 每个 thread 处理 1 个 float
// ----------------------------------------------------------------------------

__global__ void gelu_fp32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = gelu_tanh_fp32(x[idx]);
  }
}

// ----------------------------------------------------------------------------
// fp32x4 kernel
// 每个 thread 处理 4 个 float = 1 个 float4
// ----------------------------------------------------------------------------

__global__ void gelu_fp32x4_kernel(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  if (idx + 3 < N) {
    float4 vx = FLOAT4(x[idx]);
    float4 vy;
    vy.x = gelu_tanh_fp32(vx.x);
    vy.y = gelu_tanh_fp32(vx.y);
    vy.z = gelu_tanh_fp32(vx.z);
    vy.w = gelu_tanh_fp32(vx.w);
    FLOAT4(y[idx]) = vy;
  } else {
    // 处理尾部不足 4 个元素的情况
    for (int i = idx; i < N; ++i) {
      y[i] = gelu_tanh_fp32(x[i]);
    }
  }
}


// ----------------------------------------------------------------------------
// f16 scalar kernel
// 每个 thread 处理 1 个 half，内部转 fp32 计算
// ----------------------------------------------------------------------------

__global__ void gelu_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = __half2float(x[idx]);
    y[idx] = __float2half(gelu_tanh_fp32(v));
  }
}

// ----------------------------------------------------------------------------
// f16x2 kernel
// 每个 thread 处理 2 个 half = 1 个 half2
// ----------------------------------------------------------------------------

__global__ void gelu_fp16x2_kernel(half* x, half* y, int N) {
  // 当前 thread 负责第几个 half2 pack
  int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 当前 half2 pack 在原始 half 数组中的起始位置
  int base = pack_idx * 2;

  // 完整的 half2 情况：一次处理两个 half
  if (base + 1 < N) {
    float2 f = __half22float2(HALF2(x[base]));
    float2 g;
    g.x = gelu_tanh_fp32(f.x);
    g.y = gelu_tanh_fp32(f.y);
    HALF2(y[base]) = __float22half2_rn(g);
  }
  // N 为奇数时，最后剩下一个 half，单独处理
  else if (base < N) {
    float v = __half2float(x[base]);
    y[base] = __float2half(gelu_tanh_fp32(v));
  }
}

// ----------------------------------------------------------------------------
// f16x4 kernel
// 每个 thread 处理 4 个 half = 2 个 half2
// ----------------------------------------------------------------------------

__global__ void gelu_fp16x4_kernel(half* x, half* y, int N) {
  // 当前 thread 负责第几个 x4 pack
  // 一个 pack = 4 个 half = 2 个 half2
  int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 当前 x4 pack 在原始 half 数组中的起始位置
  int base = pack_idx * 4;

  // 完整的 x4 情况：一次处理 4 个 half
  if (base + 3 < N) {
    // 处理第 0、1 个 half
    float2 f0 = __half22float2(HALF2(x[base]));
    float2 g0;
    g0.x = gelu_tanh_fp32(f0.x);
    g0.y = gelu_tanh_fp32(f0.y);
    HALF2(y[base]) = __float22half2_rn(g0);

    // 处理第 2、3 个 half
    float2 f1 = __half22float2(HALF2(x[base + 2]));
    float2 g1;
    g1.x = gelu_tanh_fp32(f1.x);
    g1.y = gelu_tanh_fp32(f1.y);
    HALF2(y[base + 2]) = __float22half2_rn(g1);
  }
  // 尾部不足 4 个 half 的情况，逐个 scalar 处理，避免越界
  else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int offset = base + i;
      if (offset < N) {
        float v = __half2float(x[offset]);
        y[offset] = __float2half(gelu_tanh_fp32(v));
      }
    }
  }
}

// ----------------------------------------------------------------------------
// f16x8 kernel
// 每个 thread 处理 8 个 half = 4 个 half2
// ----------------------------------------------------------------------------

__global__ void gelu_fp16x8_kernel(half* x, half* y, int N) {
  // 当前 thread 负责第几个 x8 pack
  // 一个 pack = 8 个 half = 4 个 half2
  int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 当前 x8 pack 在原始 half 数组中的起始位置
  int base = pack_idx * 8;

  // 完整的 x8 情况：一次处理 4 个 half2
  if (base + 7 < N) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 f = __half22float2(HALF2(x[base + i * 2]));
      float2 g;
      g.x = gelu_tanh_fp32(f.x);
      g.y = gelu_tanh_fp32(f.y);
      HALF2(y[base + i * 2]) = __float22half2_rn(g);
    }
  }
  // 尾部不足 8 个 half 的情况，逐个 scalar 处理，避免越界
  else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int offset = base + i;
      if (offset < N) {
        float v = __half2float(x[offset]);
        y[offset] = __float2half(gelu_tanh_fp32(v));
      }
    }
  }
}

// ----------------------------------------------------------------------------
// f16x8 pack kernel
// 每个 thread 处理 8 个 half，并用 128-bit load/store
// ----------------------------------------------------------------------------

__global__ void gelu_fp16x8_pack_kernel(half* x, half* y, int N) {
  // 当前 thread 负责第几个 x8 pack
  // 一个 x8 pack = 8 个 half = 4 个 half2 = 16 bytes
  int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 当前 x8 pack 在原始 half 数组中的起始位置
  int base = pack_idx * 8;

  // 完整 x8 pack：使用 128-bit load/store
  if (base + 7 < N) {
    alignas(16) half pack_x[8];
    alignas(16) half pack_y[8];

    // 一次性 load 8 个 half，也就是 4 个 half2
    LDST128BITS(pack_x[0]) = LDST128BITS(x[base]);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = __half2float(pack_x[i]);
      pack_y[i] = __float2half(gelu_tanh_fp32(v));
    }

    // 一次性 store 8 个 half，也就是 4 个 half2
    LDST128BITS(y[base]) = LDST128BITS(pack_y[0]);
  }
  // 尾部不足 8 个 half：逐个 scalar 处理，避免越界
  else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int offset = base + i;
      if (offset < N) {
        float v = __half2float(x[offset]);
        y[offset] = __float2half(gelu_tanh_fp32(v));
      }
    }
  }
}


// ============================================================================
// Tensor check macros
// ============================================================================
//
// 这些宏用于在 C++/CUDA launcher 入口处检查 Python 传进来的 Tensor：
//   - 是否在 CUDA 上
//   - 是否连续存储
//   - dtype 是否符合预期
//   - shape 是否一致
//   - device 是否一致
//
// 注意：
//   kernel 内部不应该做这些检查，检查应该放在 launcher 层。
//   因为 kernel 运行在 GPU 上，错误处理和异常不方便。
// ============================================================================

//自动绑定
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

//cuda 检查宏
#define CHECK_CUDA(T)                                      \
  TORCH_CHECK((T).is_cuda(), #T " must be a CUDA tensor")

//contiguous 检查宏
#define CHECK_CONTIGUOUS(T)                                \
  TORCH_CHECK((T).is_contiguous(), #T " must be contiguous")

//dtype 检查宏
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)               \
  do {                                                     \
    if ((T).scalar_type() != (th_type)) {                  \
      std::cout << "Tensor Info: " << (T).options()        \
                << std::endl;                             \
      throw std::runtime_error("Tensor must be " #th_type);\
    }                                                      \
  } while (0)

//shape 检查宏
#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                   \
  do {                                                     \
    TORCH_CHECK((T1).dim() == (T2).dim(),                  \
                #T1 " and " #T2 " dim mismatch");         \
    for (int64_t i = 0; i < (T1).dim(); ++i) {             \
      TORCH_CHECK((T1).size(i) == (T2).size(i),            \
                  #T1 " and " #T2 " shape mismatch");     \
    }                                                      \
  } while (0)

//device 检查宏
#define CHECK_TORCH_TENSOR_DEVICE(T1, T2)                  \
  TORCH_CHECK((T1).device() == (T2).device(),              \
              #T1 " and " #T2 " must be on same device")

//组合检查宏
#define CHECK_INPUT(T)                                     \
  CHECK_CUDA(T);                                           \
  CHECK_CONTIGUOUS(T)


// ============================================================================
// Common input check
// ============================================================================
//
// GELU 是单输入 elementwise 激活算子：
//   y[i] = gelu(x[i])
//
// 要求：
//   - x、y 都必须是 CUDA tensor；
//   - x、y 都必须 contiguous；
//   - x、y 的 dtype 必须一致（fp32 对应 f32/f32x4，fp16 对应 f16 系列）；
//   - shape 必须完全一致；
//   - device 必须一致。
// ============================================================================

void check_gelu_f32_inputs(
    const torch::Tensor& x,
    const torch::Tensor& y
) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);

  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32);

  CHECK_TORCH_TENSOR_SHAPE(x, y);

  CHECK_TORCH_TENSOR_DEVICE(x, y);
}

void check_gelu_f16_inputs(
    const torch::Tensor& x,
    const torch::Tensor& y
) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);

  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat16);
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat16);

  CHECK_TORCH_TENSOR_SHAPE(x, y);

  CHECK_TORCH_TENSOR_DEVICE(x, y);
}


// ============================================================================
// GELU launch macros
// ============================================================================
//
// 这里的 gelu 是一维 flatten 算子：
//   y[i] = gelu(x[i])
//
// 和 vector_sub 类似，直接用 x.numel() 得到总元素数即可，
// 不需要按行规约。
// ============================================================================

#define GELU_THREADS 256


// ----------------------------------------------------------------------------
// scalar launch (fp32 / f16)
// 每个 thread 处理 1 个元素
// ----------------------------------------------------------------------------

#define LAUNCH_GELU_F32_KERNEL(N)                                            \
  gelu_fp32_kernel<<<grid, block>>>(                                         \
      reinterpret_cast<float*>(x.data_ptr()),                                \
      reinterpret_cast<float*>(y.data_ptr()),                                \
      (N)                                                                    \
  );

#define DISPATCH_GELU_F32_KERNEL(N)                                          \
  dim3 block(GELU_THREADS);                                                  \
  dim3 grid(((N) + GELU_THREADS - 1) / GELU_THREADS);                        \
  LAUNCH_GELU_F32_KERNEL(N)

#define LAUNCH_GELU_F16_KERNEL(N)                                            \
  gelu_f16_kernel<<<grid, block>>>(                                          \
      reinterpret_cast<half*>(x.data_ptr()),                                 \
      reinterpret_cast<half*>(y.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_GELU_F16_KERNEL(N)                                          \
  dim3 block(GELU_THREADS);                                                  \
  dim3 grid(((N) + GELU_THREADS - 1) / GELU_THREADS);                        \
  LAUNCH_GELU_F16_KERNEL(N)


// ----------------------------------------------------------------------------
// x2 / x4 / x8 / x8_pack launch
// 每个 thread 处理 PACK 个元素
// ----------------------------------------------------------------------------

#define LAUNCH_GELU_F32X4_KERNEL(N)                                          \
  gelu_fp32x4_kernel<<<grid, block>>>(                                       \
      reinterpret_cast<float*>(x.data_ptr()),                                \
      reinterpret_cast<float*>(y.data_ptr()),                                \
      (N)                                                                    \
  );

#define DISPATCH_GELU_F32X4_KERNEL(N)                                        \
  dim3 block(GELU_THREADS);                                                  \
  int packs = ((N) + 4 - 1) / 4;                                             \
  dim3 grid((packs + GELU_THREADS - 1) / GELU_THREADS);                      \
  LAUNCH_GELU_F32X4_KERNEL(N)

#define LAUNCH_GELU_F16X2_KERNEL(N)                                          \
  gelu_fp16x2_kernel<<<grid, block>>>(                                       \
      reinterpret_cast<half*>(x.data_ptr()),                                 \
      reinterpret_cast<half*>(y.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_GELU_F16X2_KERNEL(N)                                        \
  dim3 block(GELU_THREADS);                                                  \
  int packs = ((N) + 2 - 1) / 2;                                             \
  dim3 grid((packs + GELU_THREADS - 1) / GELU_THREADS);                      \
  LAUNCH_GELU_F16X2_KERNEL(N)

#define LAUNCH_GELU_F16X4_KERNEL(N)                                          \
  gelu_fp16x4_kernel<<<grid, block>>>(                                       \
      reinterpret_cast<half*>(x.data_ptr()),                                 \
      reinterpret_cast<half*>(y.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_GELU_F16X4_KERNEL(N)                                        \
  dim3 block(GELU_THREADS);                                                  \
  int packs = ((N) + 4 - 1) / 4;                                             \
  dim3 grid((packs + GELU_THREADS - 1) / GELU_THREADS);                      \
  LAUNCH_GELU_F16X4_KERNEL(N)

#define LAUNCH_GELU_F16X8_KERNEL(N)                                          \
  gelu_fp16x8_kernel<<<grid, block>>>(                                       \
      reinterpret_cast<half*>(x.data_ptr()),                                 \
      reinterpret_cast<half*>(y.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_GELU_F16X8_KERNEL(N)                                        \
  dim3 block(GELU_THREADS);                                                  \
  int packs = ((N) + 8 - 1) / 8;                                             \
  dim3 grid((packs + GELU_THREADS - 1) / GELU_THREADS);                      \
  LAUNCH_GELU_F16X8_KERNEL(N)

#define LAUNCH_GELU_F16X8_PACK_KERNEL(N)                                     \
  gelu_fp16x8_pack_kernel<<<grid, block>>>(                                  \
      reinterpret_cast<half*>(x.data_ptr()),                                 \
      reinterpret_cast<half*>(y.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_GELU_F16X8_PACK_KERNEL(N)                                   \
  dim3 block(GELU_THREADS);                                                  \
  int packs = ((N) + 8 - 1) / 8;                                             \
  dim3 grid((packs + GELU_THREADS - 1) / GELU_THREADS);                      \
  LAUNCH_GELU_F16X8_PACK_KERNEL(N)


// ============================================================================
// GELU launcher wrappers
// ============================================================================
//
// Python 侧调用形式：
//   lib.gelu_f32(x, y)
//   lib.gelu_f32x4(x, y)
//   lib.gelu_f16(x, y)
//   lib.gelu_f16x2(x, y)
//   lib.gelu_f16x4(x, y)
//   lib.gelu_f16x8(x, y)
//   lib.gelu_f16x8_pack(x, y)
//
// 这些都是 output tensor 由 Python 侧提前创建的版本。
// ============================================================================

void gelu_f32(torch::Tensor x, torch::Tensor y) {
  check_gelu_f32_inputs(x, y);

  const int N = static_cast<int>(x.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_GELU_F32_KERNEL(N)
}

void gelu_f32x4(torch::Tensor x, torch::Tensor y) {
  check_gelu_f32_inputs(x, y);

  const int N = static_cast<int>(x.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_GELU_F32X4_KERNEL(N)
}

void gelu_f16(torch::Tensor x, torch::Tensor y) {
  check_gelu_f16_inputs(x, y);

  const int N = static_cast<int>(x.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_GELU_F16_KERNEL(N)
}

void gelu_f16x2(torch::Tensor x, torch::Tensor y) {
  check_gelu_f16_inputs(x, y);

  const int N = static_cast<int>(x.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_GELU_F16X2_KERNEL(N)
}

void gelu_f16x4(torch::Tensor x, torch::Tensor y) {
  check_gelu_f16_inputs(x, y);

  const int N = static_cast<int>(x.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_GELU_F16X4_KERNEL(N)
}

void gelu_f16x8(torch::Tensor x, torch::Tensor y) {
  check_gelu_f16_inputs(x, y);

  const int N = static_cast<int>(x.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_GELU_F16X8_KERNEL(N)
}

void gelu_f16x8_pack(torch::Tensor x, torch::Tensor y) {
  check_gelu_f16_inputs(x, y);

  const int N = static_cast<int>(x.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_GELU_F16X8_PACK_KERNEL(N)
}


// ============================================================================
// Return versions
// ============================================================================
//
// Python 侧调用形式：
//   y = lib.gelu_f32_return(x)
//   y = lib.gelu_f32x4_return(x)
//   y = lib.gelu_f16_return(x)
//   y = lib.gelu_f16x2_return(x)
//   y = lib.gelu_f16x4_return(x)
//   y = lib.gelu_f16x8_return(x)
//   y = lib.gelu_f16x8_pack_return(x)
// ============================================================================

torch::Tensor gelu_f32_return(torch::Tensor x) {
  auto y = torch::empty_like(x);
  gelu_f32(x, y);
  return y;
}

torch::Tensor gelu_f32x4_return(torch::Tensor x) {
  auto y = torch::empty_like(x);
  gelu_f32x4(x, y);
  return y;
}

torch::Tensor gelu_f16_return(torch::Tensor x) {
  auto y = torch::empty_like(x);
  gelu_f16(x, y);
  return y;
}

torch::Tensor gelu_f16x2_return(torch::Tensor x) {
  auto y = torch::empty_like(x);
  gelu_f16x2(x, y);
  return y;
}

torch::Tensor gelu_f16x4_return(torch::Tensor x) {
  auto y = torch::empty_like(x);
  gelu_f16x4(x, y);
  return y;
}

torch::Tensor gelu_f16x8_return(torch::Tensor x) {
  auto y = torch::empty_like(x);
  gelu_f16x8(x, y);
  return y;
}

torch::Tensor gelu_f16x8_pack_return(torch::Tensor x) {
  auto y = torch::empty_like(x);
  gelu_f16x8_pack(x, y);
  return y;
}


// ============================================================================
// PyTorch binding
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x4)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8_pack)

  TORCH_BINDING_COMMON_EXTENSION(gelu_f32_return)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32x4_return)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16_return)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x2_return)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x4_return)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8_return)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x8_pack_return)
}
