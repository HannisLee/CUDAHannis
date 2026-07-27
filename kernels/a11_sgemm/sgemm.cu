
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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



//naive
__global__ void sgemm_v1_kernal(float *a, float *b,float *c,int M,int N,int K){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y +threadIdx.y;

    if(row>=M||col>=N)return;
    float acsum=0.0f;
    for(int i=0;i<K;i++){
        acsum+= a[row*K+i]*b[i*N+col];
    }
    c[row*N+col]=acsum;
}


#define TILE_SIZE 16

__global__ void sgemm_v2_kernel(float *a, float *b, float *c, int M, int N, int K) {
    //输出C的位置
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
	//block内坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float acsum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
        int a_col = k0 + tx;
        int b_row = k0 + ty;

        if (row < M && a_col < K) {
            As[ty][tx] = a[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[ty][tx] = b[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acsum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = acsum;
    }
}


// ============================================================================
// Host Wrappers(核函数未改动,仅补充 host 端封装与 pybind 注册)
// C = A @ B,A: M×K,B: K×N,C: M×N
// ============================================================================

void sgemm_v1(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K) {
    // v1: naive,每个线程计算 C 的一个元素;2D block (16,16)
    int bsx = 16;
    int bsy = 16;
    dim3 block(bsx, bsy);
    dim3 grid((N + bsx - 1) / bsx, (M + bsy - 1) / bsy);
    sgemm_v1_kernal<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
}

void sgemm_v2(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K) {
    // v2: 共享内存分块;block (TILE_SIZE, TILE_SIZE)
    constexpr int TILE = TILE_SIZE;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_v2_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
}


// ============================================================================
// v5: cuBLAS 版本
// 行优先 C = A@B  等价于  列优先 C^T = B^T @ A^T:
//   直接把行优先的 B 当作列优先 B^T、A 当作 A^T、C 当作 C^T 传入即可。
// ============================================================================

static cublasHandle_t get_cublas_handle() {
    // handle 创建开销大,进程级单例 lazy init
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        cublasCreate_v2(&h);
        return h;
    }();
    return handle;
}

void sgemm_v5(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    // op(A)=B^T (N×K, lda=N), op(B)=A^T (K×M, ldb=K), 结果 C=C^T (N×M, ldc=N)
    cublasSgemm_v2(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b.data_ptr<float>(), N,
        a.data_ptr<float>(), K,
        &beta,
        c.data_ptr<float>(), N);
}


// ============================================================================
// Pybind 自动绑定
// ============================================================================

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sgemm_v1);
  TORCH_BINDING_COMMON_EXTENSION(sgemm_v2);
  TORCH_BINDING_COMMON_EXTENSION(sgemm_v5);
}

