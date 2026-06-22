#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void vector_add_kernel(const scalar_t* __restrict__ x,
                                  const scalar_t* __restrict__ y,
                                  scalar_t* __restrict__ out,
                                  int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + y[idx];
  }
}

torch::Tensor cuda_add_forward(torch::Tensor x, torch::Tensor y) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must have the same dtype");

  auto out = torch::empty_like(x);
  const auto n = out.numel();
  if (n == 0) {
    return out;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "cuda_add_forward", [&] {
    vector_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
