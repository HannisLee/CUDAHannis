
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



//naive
__global__ void sgemm_v1_kernal(float *a, float *b,float *c,int M,int N,int K){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.t +threadIdx.y;

    if(row>=M||col>=N)return;
    float acsum=0.0f;
    for(int i=0;i<K;i++){
        acsum+= A[row*K+i]*B[i*N+col];
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

//启动
dim3 block(TILE_SIZE, TILE_SIZE);
dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
          (M + TILE_SIZE - 1) / TILE_SIZE);