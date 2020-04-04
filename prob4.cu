#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#define MATRIX_DIM 500
#define MATRIX_SIZE (MATRIX_DIM * MATRIX_DIM)
#define BLOCK_WIDTH 16
#define TILE_WIDTH 4
void mulWithCuda(float *p, const float *m, const float *n);
// calculate one element with one thread
__global__ void matrixMultiplication(float *P, float *M, float *N)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float p_val = 0;
    for (int ph = 0; ph < MATRIX_DIM / TILE_WIDTH; ph++)
    {
        Mds[ty][tx] = M[row*MATRIX_DIM + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*MATRIX_DIM + col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            p_val += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[row*MATRIX_DIM + col] = p_val;
}
void MatrixMulCPU(float *P, const float *M, const float *N)
{
    cudaEvent_t start, stop;
    float cpu_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); // start timer
    for (int i = 0; i < MATRIX_DIM; i++)
    {
        for (int j = 0; j < MATRIX_DIM; j++)
        {
            float p_val = 0;
            for (int k = 0; k < MATRIX_DIM; k++)
                p_val += M[i * MATRIX_DIM + k] * N[k * MATRIX_DIM + j];
            P[i * MATRIX_DIM + j] = p_val;
        }
    }
    cudaEventRecord(stop, 0);    // end timer and display results
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);
    printf("CPU time (ms):\t%.2f\n", cpu_time);
}
void verifyGPUsoln(const float *GPU_P, const float *M, const float *N)
{
    bool passed = true;
    float *P = (float *)malloc(MATRIX_SIZE * sizeof(float));
    MatrixMulCPU(P, M, N);
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        if (GPU_P[i] != P[i])
        {
            passed = false;
            break;
        }
    }
    if (passed)    printf("TEST PASSED\n");
    else        printf("TEST FAILED\n");
}
　
int main()
{
    float *m = (float *)malloc(MATRIX_SIZE * sizeof(float)); // yeah, there ain't enough room on the
    float *n = (float *)malloc(MATRIX_SIZE * sizeof(float)); // stack for 3 5000x5000 matricies
    float *p = (float *)malloc(MATRIX_SIZE * sizeof(float)); // you can mess with VS's settings but I'd rather not.
    for (int i = 0; i < MATRIX_SIZE; i++)
    {    // value between 0 and 10, one decimal place
        m[i] = rand() % 100 / 10.0;
        n[i] = rand() % 100 / 10.0;
    }
    mulWithCuda(p, m, n);
    verifyGPUsoln(p, m, n);
    free(m);
    free(n);
    free(p);
    return 0;
}
// Helper function for using CUDA to add vectors in parallel.
void mulWithCuda(float *p, const float *m, const float *n)
{
    float *dev_m = 0;
    float *dev_n = 0;
    float *dev_p = 0;
    cudaError_t malloc_test;
    // Allocate GPU buffers for three vectors (two input, one output)
    malloc_test = cudaMalloc((void**)&dev_p, MATRIX_SIZE * sizeof(float));    // p
    if (malloc_test != cudaSuccess) printf("error allocating mem for dev_p\n");
    malloc_test = cudaMalloc((void**)&dev_m, MATRIX_SIZE * sizeof(float));    // m
    if (malloc_test != cudaSuccess) printf("error allocating mem for dev_m\n");
    malloc_test = cudaMalloc((void**)&dev_n, MATRIX_SIZE * sizeof(float));    // n
    if (malloc_test != cudaSuccess) printf("error allocating mem for dev_n\n");
    // Copy input vectors from host memory to GPU.
    cudaMemcpy(dev_m, m, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_n, n, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
　
    // create event-based timers
    cudaEvent_t start, stop;
    float gpu_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // general error for checking errors during GPU processing
    cudaError_t mulErr;
　
    int numBlocks = MATRIX_DIM / TILE_WIDTH;
    if (MATRIX_DIM % TILE_WIDTH) numBlocks++;
    dim3 grid(numBlocks, numBlocks);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    cudaEventRecord(start, 0); // start timer
    matrixMultiplication << <grid, block >> >(dev_p, dev_m, dev_n);
    mulErr = cudaGetLastError();
    if (mulErr != cudaSuccess) printf("Error during multiplication: %s", cudaGetErrorString(mulErr));
    cudaEventRecord(stop, 0);    // end timer and display results
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU time (ms):\t%.2f\n", gpu_time); //TODO - add output of block size to make excel easier
    cudaDeviceSynchronize();
    // Copy output vector from GPU buffer to host memory.
    cudaError_t cpyErr = cudaMemcpy(p, dev_p, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (cpyErr != cudaSuccess) printf("error copying dev_p to host\n");
    cudaError_t freeErr;
    freeErr = cudaFree(dev_p);
    if (freeErr != cudaSuccess) printf("error freeing dev_p\n");    // p
    freeErr = cudaFree(dev_m);
    if (freeErr != cudaSuccess) printf("error freeing dev_m\n");    // m
    freeErr = cudaFree(dev_n);
    if (freeErr != cudaSuccess) printf("error freeing dev_n\n");    // n
}

