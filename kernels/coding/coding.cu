#include <algorithm>
#include <cmath>
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


#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void coding1_kernel(float *a,float *b,int N){
    








    
}









void coding1(torch::Tensor a,torch::Tensor b) {
    int n = a.numel();

    if (n > 0) {
        int threads = BLOCK_SIZE;
        int blocks = (n + threads - 1) / threads;

        coding1_kernel<<<blocks, threads>>>(
            a.data_ptr<float>(),b.data_ptr<float>(),n);
    }
}




//自动绑定
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(coding1);
}


