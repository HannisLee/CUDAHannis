#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Half.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

__device__ __forceinline__ uint8_t e4m3_encode_magnitude(float value) {
  value = fminf(fmaxf(value, 0.0f), 448.0f);
  if (value < 0.015625f) {
    int mant = static_cast<int>(floorf(value * 512.0f + 0.5f));
    mant = mant < 0 ? 0 : mant;
    mant = mant > 7 ? 7 : mant;
    return static_cast<uint8_t>(mant);
  }

  float exp_unbiased_f = floorf(log2f(fmaxf(value, 0.015625f)));
  float base = exp2f(exp_unbiased_f);
  int mant = static_cast<int>(floorf((value / base - 1.0f) * 8.0f + 0.5f));
  int exp_unbiased = static_cast<int>(exp_unbiased_f);
  if (mant >= 8) {
    exp_unbiased += 1;
    mant = 0;
  }

  int exp_biased = exp_unbiased + 7;
  if (exp_biased > 15) {
    exp_biased = 15;
    mant = 6;
  }
  if (exp_biased < 1) {
    exp_biased = 1;
  }
  mant = mant < 0 ? 0 : mant;
  mant = mant > 7 ? 7 : mant;
  if (exp_biased == 15 && mant > 6) {
    mant = 6;
  }
  return static_cast<uint8_t>((exp_biased << 3) | mant);
}

__device__ __forceinline__ float e4m3_decode(uint8_t code) {
  bool negative = (code & 0x80) != 0;
  uint8_t mag = code & 0x7F;
  int exp = (mag >> 3) & 0x0F;
  int mant = mag & 0x07;
  if (exp == 15 && mant == 7) {
    mant = 6;
  }

  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0.001953125f;
  } else {
    value = exp2f(static_cast<float>(exp) - 7.0f) * (1.0f + static_cast<float>(mant) * 0.125f);
  }
  return negative ? -value : value;
}

__global__ void quantize_e4m3_per_row_kernel(const c10::Half* __restrict__ x,
                                             uint8_t* __restrict__ codes,
                                             float* __restrict__ scales,
                                             int64_t rows,
                                             int head_dim) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float reduction[128];
  float local_max = 0.0f;
  if (tid < head_dim) {
    float value = static_cast<float>(x[row * head_dim + tid]);
    local_max = fabsf(value);
  }
  reduction[tid] = local_max;
  __syncthreads();

  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }

  float scale = reduction[0] > 0.0f ? reduction[0] / 448.0f : 1.0f;
  if (tid == 0) {
    scales[row] = scale;
  }
  if (tid < head_dim) {
    float value = static_cast<float>(x[row * head_dim + tid]);
    float normalized = fminf(fabsf(value) / scale, 448.0f);
    uint8_t code = e4m3_encode_magnitude(normalized);
    if (value < 0.0f) {
      code |= 0x80;
    }
    codes[row * head_dim + tid] = code;
  }
}

__device__ __forceinline__ float dot_qk(const uint8_t* __restrict__ q_codes,
                                        const uint8_t* __restrict__ k_codes,
                                        const float* __restrict__ q_scales,
                                        const float* __restrict__ k_scales,
                                        int64_t q_row,
                                        int64_t k_row,
                                        int head_dim) {
  float q_scale = q_scales[q_row];
  float k_scale = k_scales[k_row];
  float sum = 0.0f;
  int64_t q_base = q_row * head_dim;
  int64_t k_base = k_row * head_dim;
#pragma unroll
  for (int d = 0; d < 128; ++d) {
    if (d < head_dim) {
      float q = e4m3_decode(q_codes[q_base + d]) * q_scale;
      float k = e4m3_decode(k_codes[k_base + d]) * k_scale;
      sum += q * k;
    }
  }
  return sum;
}

__global__ void flash_attention_fp8_kernel(const uint8_t* __restrict__ q_codes,
                                           const uint8_t* __restrict__ k_codes,
                                           const uint8_t* __restrict__ v_codes,
                                           const float* __restrict__ q_scales,
                                           const float* __restrict__ k_scales,
                                           const float* __restrict__ v_scales,
                                           c10::Half* __restrict__ out,
                                           int64_t rows,
                                           int seq_len,
                                           int head_dim,
                                           float sm_scale) {
  int64_t q_row = blockIdx.x;
  int tid = threadIdx.x;
  if (q_row >= rows) {
    return;
  }

  extern __shared__ __half scores[];
  __shared__ float reduction[256];
  __shared__ float row_max;
  __shared__ float row_sum;

  int query_pos = static_cast<int>(q_row % seq_len);
  int64_t bh_base = q_row - query_pos;

  float local_max = -3.4028234663852886e38f;
  for (int key_pos = tid; key_pos <= query_pos; key_pos += blockDim.x) {
    int64_t k_row = bh_base + key_pos;
    float score = dot_qk(q_codes, k_codes, q_scales, k_scales, q_row, k_row, head_dim) * sm_scale;
    __half score_h = __float2half(score);
    scores[key_pos] = score_h;
    local_max = fmaxf(local_max, __half2float(score_h));
  }

  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    row_max = reduction[0];
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int key_pos = tid; key_pos <= query_pos; key_pos += blockDim.x) {
    local_sum += expf(__half2float(scores[key_pos]) - row_max);
  }
  reduction[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    row_sum = reduction[0];
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int key_pos = 0; key_pos <= query_pos; ++key_pos) {
      int64_t v_row = bh_base + key_pos;
      float p = expf(__half2float(scores[key_pos]) - row_max) / row_sum;
      float v = e4m3_decode(v_codes[v_row * head_dim + d]) * v_scales[v_row];
      acc += p * v;
    }
    out[q_row * head_dim + d] = c10::Half(acc);
  }
}

torch::Tensor flash_attention_fp8_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v must have the same shape");
  TORCH_CHECK(q.dim() == 4, "q, k, v must have shape (B, H, S, D)");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16,
              "flash_attention_fp8 CUDA supports float16 inputs");

  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int seq_len = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));
  TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
  TORCH_CHECK(seq_len > 0, "sequence length must be non-empty");
  TORCH_CHECK(seq_len <= 8192, "CUDA implementation supports sequence length <= 8192");

  const int64_t rows = batch * heads * seq_len;
  auto code_options = q.options().dtype(torch::kUInt8);
  auto scale_options = q.options().dtype(torch::kFloat32);
  auto q_codes = torch::empty_like(q, code_options);
  auto k_codes = torch::empty_like(k, code_options);
  auto v_codes = torch::empty_like(v, code_options);
  auto q_scales = torch::empty({batch, heads, seq_len}, scale_options);
  auto k_scales = torch::empty({batch, heads, seq_len}, scale_options);
  auto v_scales = torch::empty({batch, heads, seq_len}, scale_options);
  auto out = torch::empty_like(q);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int quant_threads = 128;
  quantize_e4m3_per_row_kernel<<<static_cast<int>(rows), quant_threads, 0, stream>>>(
      q.data_ptr<c10::Half>(), q_codes.data_ptr<uint8_t>(), q_scales.data_ptr<float>(), rows, head_dim);
  quantize_e4m3_per_row_kernel<<<static_cast<int>(rows), quant_threads, 0, stream>>>(
      k.data_ptr<c10::Half>(), k_codes.data_ptr<uint8_t>(), k_scales.data_ptr<float>(), rows, head_dim);
  quantize_e4m3_per_row_kernel<<<static_cast<int>(rows), quant_threads, 0, stream>>>(
      v.data_ptr<c10::Half>(), v_codes.data_ptr<uint8_t>(), v_scales.data_ptr<float>(), rows, head_dim);

  constexpr int attn_threads = 256;
  size_t shared_bytes = static_cast<size_t>(seq_len) * sizeof(__half);
  flash_attention_fp8_kernel<<<static_cast<int>(rows), attn_threads, shared_bytes, stream>>>(
      q_codes.data_ptr<uint8_t>(),
      k_codes.data_ptr<uint8_t>(),
      v_codes.data_ptr<uint8_t>(),
      q_scales.data_ptr<float>(),
      k_scales.data_ptr<float>(),
      v_scales.data_ptr<float>(),
      out.data_ptr<c10::Half>(),
      rows,
      seq_len,
      head_dim,
      static_cast<float>(sm_scale));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
