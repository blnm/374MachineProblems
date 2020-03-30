#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define MATRIX_DIM 500
#define MATRIX_SIZE (MATRIX_DIM * MATRIX_DIM)
#define BLOCK_WIDTH 16

void mulWithCuda(float *p, const float *m, const float *n);
void testTransferTime(float *m, float *n);

// calculate one element with one thread
__global__ void matrixMultiplication(float *P, float *M, float *N)
{
	// calculate row, col index
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < MATRIX_DIM && col < MATRIX_DIM)
	{
		float p_val = 0;

		// do multiplication here

		P[row*MATRIX_DIM + col] = p_val; // M[ind] + N[ind];
	}
	__syncthreads();
}

void verifyGPUsoln(const float *GPU_C, const float *A, const float *B)
{
	bool passed = true;
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		if (GPU_C[i] != A[i] + B[i])
		{
			passed = false;
			break;
		}
	}

	if (passed)	printf("TEST PASSED\n");
	else		printf("TEST FAILED\n");
}


int main()
{
	float *a = (float *)malloc(MATRIX_SIZE * sizeof(float)); // yeah, there ain't enough room on the
	float *b = (float *)malloc(MATRIX_SIZE * sizeof(float)); // stack for 3 5000x5000 matricies
	float *c = (float *)malloc(MATRIX_SIZE * sizeof(float)); // you can mess with VS's settings but I'd rather not.

	for (int i = 0; i < MATRIX_SIZE; i++)
	{	// value between 0 and 10, one decimal place
		a[i] = rand() % 100 / 10.0;
		b[i] = rand() % 100 / 10.0;
	}

	testTransferTime(a, b);

	//mulWithCuda(p, m, n);

	//verifyGPUsoln(p, m, n);

	free(a);
	free(b);
	free(c);

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
	malloc_test = cudaMalloc((void**)&dev_p, MATRIX_SIZE * sizeof(float));	// p
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_p\n");

	malloc_test = cudaMalloc((void**)&dev_m, MATRIX_SIZE * sizeof(float));	// m
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_m\n");

	malloc_test = cudaMalloc((void**)&dev_n, MATRIX_SIZE * sizeof(float));	// n
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_n\n");

	// Copy input vectors from host memory to GPU.
	cudaMemcpy(dev_m, m, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_n, n, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);


	// create event-based timers
	cudaEvent_t start, stop;
	float gpu_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// generall error for checking errors during GPU processing
	cudaError_t mulErr;


	// create block/thread dims with add-by-element as default
	int numBlocks = MATRIX_DIM / BLOCK_WIDTH;
	if (MATRIX_DIM % BLOCK_WIDTH) numBlocks++;
	dim3 grid(numBlocks, numBlocks);
	dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

	cudaEventRecord(start, 0); // start timer

	matrixMultiplication <<<grid, block >>>(dev_p, dev_m, dev_n);
	mulErr = cudaGetLastError();
	if (mulErr != cudaSuccess) printf("Error during addition: %s", cudaGetErrorString(addErr));


	cudaEventRecord(stop, 0);	// end timer and display results
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);

	printf("time taken (ms):\t%.2f\n", gpu_time); //TODO - add output of block size to make excel easier


	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaError_t cpyErr = cudaMemcpy(p, dev_p, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	if (cpyErr != cudaSuccess) printf("error copying dev_p to host\n");

	cudaError_t freeErr;
	freeErr = cudaFree(dev_p);
	if (freeErr != cudaSuccess) printf("error freeing dev_p\n");	// p

	freeErr = cudaFree(dev_m);
	if (freeErr != cudaSuccess) printf("error freeing dev_m\n");	// m

	freeErr = cudaFree(dev_n);
	if (freeErr != cudaSuccess) printf("error freeing dev_n\n");	// n
}


void testTransferTime(float *m, float *n)
{
	float *dev_m = 0;
	float *dev_n = 0;

	cudaError_t malloc_test;

	// Allocate GPU buffers for two vectors
	malloc_test = cudaMalloc((void**)&dev_m, MATRIX_SIZE * sizeof(float));	// m
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_m\n");

	malloc_test = cudaMalloc((void**)&dev_n, MATRIX_SIZE * sizeof(float));	// n
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_n\n");

	// create event-based timers
	cudaEvent_t start, stop;
	float gpu_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0); // start timer

							   // Copy from host memory to GPU.
	cudaMemcpy(dev_m, m, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_n, n, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(stop, 0);	// end timer and display results
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("copy host to device (ms):\t%.2f\n", gpu_time);

	cudaEventRecord(start, 0); // start timer

							   // Copy from GPU buffer to host memory.
	cudaMemcpy(m, dev_m, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(n, dev_n, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);	// end timer and display results
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("copy device to host (ms):\t%.2f\n", gpu_time);

	cudaError_t freeErr;
	freeErr = cudaFree(dev_m);
	if (freeErr != cudaSuccess) printf("error freeing dev_p\n");	// m

	freeErr = cudaFree(dev_n);
	if (freeErr != cudaSuccess) printf("error freeing dev_m\n");	// n
}