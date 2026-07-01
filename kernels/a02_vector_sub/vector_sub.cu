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


__global__ void vector_sub_fp32_kernal(float *a,float *b,float* c,int N){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid*blockDim.x+tid;
  if(idx<N){
    c[idx]=a[idx]-b[idx];
  }
}

__global__ void vector_sub_fp32x4_kernal(float *a,float *b,float *c,int N){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = (blockDim.x*blockIdx.x+threadIdx.x)*4;
  
  if (idx + 3 < N) {
        float4 va = FLOAT4(a[idx]);
        float4 vb = FLOAT4(b[idx]);

        float4 vc;
        vc.x = va.x - vb.x;
        vc.y = va.y - vb.y;
        vc.z = va.z - vb.z;
        vc.w = va.w - vb.w;

        FLOAT4(c[idx]) = vc;
    } else {
        // 处理尾部不足 4 个元素的情况
        for (int i = idx; i < N; ++i) {
            c[i] = a[i] - b[i];
        }
    }
}

__global__ void vector_sub_fp32x4_pack_kernal(float *a,float *b,float *c,int N){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = (blockDim.x*blockIdx.x+threadIdx.x)*4;.
  if (idx + 3 < N) {
    float4 va = LDST128BITS(a[idx]);
    float4 vb = LDST128BITS(b[idx]);
    float4 vc;
    vc.x = va.x - vb.x;
    vc.y = va.y - vb.y;
    vc.z = va.z - vb.z;
    vc.w = va.w - vb.w;
    LDST128BITS(c[idx])=LDST128BITS(vc);
    } else {
        // 处理尾部不足 4 个元素的情况
        for (int i = idx; i < N; ++i) {
            c[i] = a[i] - b[i];
        }
    }
}


__global__ void vector_sub_kernal(half *a, half *b, half *c, int N) {
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x;  // 0..N-1
    int idx = bid * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void vector_sub_fp16x2_kernel(
    half* a,
    half* b,
    half* c,
    int N
) {
    // 当前 thread 负责第几个 half2 pack
    int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前 half2 pack 在原始 half 数组中的起始位置
    int base = pack_idx * 2;

    // 完整的 half2 情况：一次处理两个 half
    if (base + 1 < N) {
        half2 a2 = HALF2(a[base]);
        half2 b2 = HALF2(b[base]);
        HALF2(c[base]) = __hsub2(a2, b2);
    }
    // N 为奇数时，最后剩下一个 half，单独处理
    else if (base < N) {
        c[base] = __hsub(a[base], b[base]);
    }
}

__global__ void vector_sub_fp16x4_kernel(
    half* a,
    half* b,
    half* c,
    int N
) {
    // 当前 thread 负责第几个 x4 pack
    // 一个 pack = 4 个 half = 2 个 half2
    int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前 x4 pack 在原始 half 数组中的起始位置
    int base = pack_idx * 4;

    // 完整的 x4 情况：一次处理 4 个 half
    if (base + 3 < N) {
        // 处理第 0、1 个 half
        half2 a2_0 = HALF2(a[base]);
        half2 b2_0 = HALF2(b[base]);
        HALF2(c[base]) = __hsub2(a2_0, b2_0);

        // 处理第 2、3 个 half
        half2 a2_1 = HALF2(a[base + 2]);
        half2 b2_1 = HALF2(b[base + 2]);
        HALF2(c[base + 2]) = __hsub2(a2_1, b2_1);
    }
    // 尾部不足 4 个 half 的情况，逐个 scalar 处理，避免越界
    else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int offset = base + i;
            if (offset < N) {
                c[offset] = __hsub(a[offset], b[offset]);
            }
        }
    }
}

__global__ void vector_sub_fp16x8_kernel(
    half* a,
    half* b,
    half* c,
    int N
) {
    // 当前 thread 负责第几个 x8 pack
    // 一个 pack = 8 个 half = 4 个 half2
    int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前 x8 pack 在原始 half 数组中的起始位置
    int base = pack_idx * 8;

    // 完整的 x8 情况：一次处理 4 个 half2
    if (base + 7 < N) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            half2 a2 = HALF2(a[base + i * 2]);
            half2 b2 = HALF2(b[base + i * 2]);
            HALF2(c[base + i * 2]) = __hsub2(a2, b2);
        }
    }
    // 尾部不足 4 个 half 的情况，逐个 scalar 处理，避免越界
    else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int offset = base + i;
            if (offset < N) {
                c[offset] = __hsub(a[offset], b[offset]);
            }
        }
    }
}

__global__ void vector_sub_fp16x8_pack_kernel(
    half* a,
    half* b,
    half* c,
    int N
) {
    // 当前 thread 负责第几个 x8 pack
    // 一个 x8 pack = 8 个 half = 4 个 half2 = 16 bytes
    int pack_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前 x8 pack 在原始 half 数组中的起始位置
    int base = pack_idx * 8;

    // 完整 x8 pack：使用 128-bit load/store
    if (base + 7 < N) {
        alignas(16) half2 pack_a[4];
        alignas(16) half2 pack_b[4];
        alignas(16) half2 pack_c[4];

        // 一次性 load 8 个 half，也就是 4 个 half2
        LDST128BITS(pack_a[0]) = LDST128BITS(a[base]);
        LDST128BITS(pack_b[0]) = LDST128BITS(b[base]);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            // 每次处理一个 half2，也就是 2 个 half
            pack_c[i] = __hsub2(pack_a[i], pack_b[i]);
        }

        // 一次性 store 8 个 half，也就是 4 个 half2
        LDST128BITS(c[base]) = LDST128BITS(pack_c[0]);
    }
    // 尾部不足 8 个 half：逐个 scalar 处理，避免越界
    else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int offset = base + i;
            if (offset < N) {
                c[offset] = __hsub(a[offset], b[offset]);
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
//   - dtype 是否为 torch.float16
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

void check_vector_sub_f16_inputs(
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
// VectorSub launch macros
// ============================================================================
//
// 这里的 vector_sub 是一维 flatten 算子：
//   c[i] = a[i] - b[i]
//
// 和 RMSNorm 不同：
//   RMSNorm 通常把输入看成 [N, K]，一个 block 处理一行。
//   vector_sub 不需要按行规约，所以直接用 a.numel() 得到总元素数即可。
// ============================================================================

#define VECTOR_SUB_THREADS 256


// ----------------------------------------------------------------------------
// scalar f16 launch
// 每个 thread 处理 1 个 half
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_SUB_F16_KERNEL(N)                                      \
  vector_sub_kernel<<<grid, block>>>(                                        \
      reinterpret_cast<half*>(a.data_ptr()),                                 \
      reinterpret_cast<half*>(b.data_ptr()),                                 \
      reinterpret_cast<half*>(c.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_VECTOR_SUB_F16_KERNEL(N)                                    \
  dim3 block(VECTOR_SUB_THREADS);                                            \
  dim3 grid(((N) + VECTOR_SUB_THREADS - 1) / VECTOR_SUB_THREADS);            \
  LAUNCH_VECTOR_SUB_F16_KERNEL(N)


// ----------------------------------------------------------------------------
// f16x2 launch
// 每个 thread 处理 2 个 half = 1 个 half2
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_SUB_F16X2_KERNEL(N)                                    \
  vector_sub_fp16x2_kernel<<<grid, block>>>(                                 \
      reinterpret_cast<half*>(a.data_ptr()),                                 \
      reinterpret_cast<half*>(b.data_ptr()),                                 \
      reinterpret_cast<half*>(c.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_VECTOR_SUB_F16X2_KERNEL(N)                                  \
  dim3 block(VECTOR_SUB_THREADS);                                            \
  int packs = ((N) + 2 - 1) / 2;                                             \
  dim3 grid((packs + VECTOR_SUB_THREADS - 1) / VECTOR_SUB_THREADS);          \
  LAUNCH_VECTOR_SUB_F16X2_KERNEL(N)


// ----------------------------------------------------------------------------
// f16x4 launch
// 每个 thread 处理 4 个 half = 2 个 half2
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_SUB_F16X4_KERNEL(N)                                    \
  vector_sub_fp16x4_kernel<<<grid, block>>>(                                 \
      reinterpret_cast<half*>(a.data_ptr()),                                 \
      reinterpret_cast<half*>(b.data_ptr()),                                 \
      reinterpret_cast<half*>(c.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_VECTOR_SUB_F16X4_KERNEL(N)                                  \
  dim3 block(VECTOR_SUB_THREADS);                                            \
  int packs = ((N) + 4 - 1) / 4;                                             \
  dim3 grid((packs + VECTOR_SUB_THREADS - 1) / VECTOR_SUB_THREADS);          \
  LAUNCH_VECTOR_SUB_F16X4_KERNEL(N)


// ----------------------------------------------------------------------------
// f16x8 launch
// 每个 thread 处理 8 个 half = 4 个 half2
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_SUB_F16X8_KERNEL(N)                                    \
  vector_sub_fp16x8_kernel<<<grid, block>>>(                                 \
      reinterpret_cast<half*>(a.data_ptr()),                                 \
      reinterpret_cast<half*>(b.data_ptr()),                                 \
      reinterpret_cast<half*>(c.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_VECTOR_SUB_F16X8_KERNEL(N)                                  \
  dim3 block(VECTOR_SUB_THREADS);                                            \
  int packs = ((N) + 8 - 1) / 8;                                             \
  dim3 grid((packs + VECTOR_SUB_THREADS - 1) / VECTOR_SUB_THREADS);          \
  LAUNCH_VECTOR_SUB_F16X8_KERNEL(N)


// ----------------------------------------------------------------------------
// f16x8 pack launch
// 每个 thread 处理 8 个 half，并用 128-bit load/store
// ----------------------------------------------------------------------------

#define LAUNCH_VECTOR_SUB_F16X8_PACK_KERNEL(N)                               \
  vector_sub_fp16x8_pack_kernel<<<grid, block>>>(                            \
      reinterpret_cast<half*>(a.data_ptr()),                                 \
      reinterpret_cast<half*>(b.data_ptr()),                                 \
      reinterpret_cast<half*>(c.data_ptr()),                                 \
      (N)                                                                    \
  );

#define DISPATCH_VECTOR_SUB_F16X8_PACK_KERNEL(N)                             \
  dim3 block(VECTOR_SUB_THREADS);                                            \
  int packs = ((N) + 8 - 1) / 8;                                             \
  dim3 grid((packs + VECTOR_SUB_THREADS - 1) / VECTOR_SUB_THREADS);          \
  LAUNCH_VECTOR_SUB_F16X8_PACK_KERNEL(N)


// ============================================================================
// VectorSub launcher wrappers
// ============================================================================
//
// Python 侧调用形式：
//   lib.vector_sub_f16(a, b, c)
//   lib.vector_sub_f16x2(a, b, c)
//   lib.vector_sub_f16x4(a, b, c)
//   lib.vector_sub_f16x8(a, b, c)
//   lib.vector_sub_f16x8_pack(a, b, c)
//
// 这些都是 output tensor 由 Python 侧提前创建的版本。
// ============================================================================

void vector_sub_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check_vector_sub_f16_inputs(a, b, c);

  const int N = static_cast<int>(a.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_VECTOR_SUB_F16_KERNEL(N)
}

void vector_sub_f16x2(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check_vector_sub_f16_inputs(a, b, c);

  const int N = static_cast<int>(a.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_VECTOR_SUB_F16X2_KERNEL(N)
}

void vector_sub_f16x4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check_vector_sub_f16_inputs(a, b, c);

  const int N = static_cast<int>(a.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_VECTOR_SUB_F16X4_KERNEL(N)
}

void vector_sub_f16x8(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check_vector_sub_f16_inputs(a, b, c);

  const int N = static_cast<int>(a.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_VECTOR_SUB_F16X8_KERNEL(N)
}

void vector_sub_f16x8_pack(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check_vector_sub_f16_inputs(a, b, c);

  const int N = static_cast<int>(a.numel());
  if (N == 0) {
    return;
  }

  DISPATCH_VECTOR_SUB_F16X8_PACK_KERNEL(N)
}


// ============================================================================
// Return versions
// ============================================================================
//
// Python 侧调用形式：
//   c = lib.vector_sub_f16_return(a, b)
//   c = lib.vector_sub_f16x2_return(a, b)
//   c = lib.vector_sub_f16x4_return(a, b)
//   c = lib.vector_sub_f16x8_return(a, b)
//   c = lib.vector_sub_f16x8_pack_return(a, b)
// ============================================================================

torch::Tensor vector_sub_f16_return(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty_like(a);
  vector_sub_f16(a, b, c);
  return c;
}

torch::Tensor vector_sub_f16x2_return(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty_like(a);
  vector_sub_f16x2(a, b, c);
  return c;
}

torch::Tensor vector_sub_f16x4_return(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty_like(a);
  vector_sub_f16x4(a, b, c);
  return c;
}

torch::Tensor vector_sub_f16x8_return(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty_like(a);
  vector_sub_f16x8(a, b, c);
  return c;
}

torch::Tensor vector_sub_f16x8_pack_return(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty_like(a);
  vector_sub_f16x8_pack(a, b, c);
  return c;
}


// ============================================================================
// PyTorch binding
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16)
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16x4)
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16x8_pack)

  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16_return)
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16x2_return)
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16x4_return)
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16x8_return)
  TORCH_BINDING_COMMON_EXTENSION(vector_sub_f16x8_pack_return)
}
