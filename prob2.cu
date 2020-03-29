#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define MATRIX_DIM 5000
#define MATRIX_SIZE (MATRIX_DIM * MATRIX_DIM)
#define BLOCK_WIDTH 16

void addWithCuda(float *c, const float *a, const float *b);

// assign
__global__ void matrixAddition(float *C, float *A, float *B)
{
	// calculate row, col index
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < MATRIX_DIM && col < MATRIX_DIM)
	{
		float c_val = 0;
		int ind = row*MATRIX_DIM + col;

		C[ind] = A[ind] + B[ind];
		//if (C[ind] == 0) printf("wtf");
		/*if (ind == 1146177)
		{
			printf("%f, %f, %f\n", A[ind], B[ind], C[ind]);
		}*/
		//printf("%f, %f, %f\n", A[ind], B[ind], C[ind]);
	}
	__syncthreads();
}

bool verifyGPUsoln(const float *GPU_C, const float *A, const float *B)
{
	/*float C[MATRIX_SIZE] = {};

	for (int i = 0; i < MATRIX_DIM; i++)
	{
		for (int j = 0; j < MATRIX_DIM; j++)
			C[i*MATRIX_DIM + j] = A[i*MATRIX_DIM + j] + B[i*MATRIX_DIM + j];
	}*/
	bool passed = true;
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		float c = A[i] + B[i];
		// everything before index 10000 is incorrect for some reason.
		if (GPU_C[i] != c)//C[i])
		{
			printf("%d\t%f\t%f\t%f\t%f\n", i, A[i], B[i], GPU_C[i], c);
			passed = false;
			break;
		}
	}

	//return passed;

	if (passed)
	{
		printf("TEST PASSED\n");
	}
	else
		printf("TEST FAILED\n");

	return passed;
}


int main()
{
	/*float a[MATRIX_SIZE] = {};
	float b[MATRIX_SIZE] = {};
	float c[MATRIX_SIZE] = {};
	*/
	float *a = (float *)malloc(MATRIX_SIZE * sizeof(float));
	float *b = (float *)malloc(MATRIX_SIZE * sizeof(float));
	float *c = (float *)malloc(MATRIX_SIZE * sizeof(float));
	
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		a[i] = rand() % 100 / 10; // value between 0 and 10, one decimal place
		b[i] = rand() % 100 / 10; // value between 0 and 10
	}
		

	printf("starting addition\n");
	addWithCuda(c, a, b);
	//printf("%f\n", c[10]);
	printf("addition finished\n");

	bool passed = verifyGPUsoln(c, a, b);
	//if (passed) printf("passed\n");
	//else printf("failed\n");

	free(a);
	free(b);
	free(c);
	
	
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(float *c, const float *a, const float *b)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;

	cudaError_t malloc_test;

	// Allocate GPU buffers for three vectors (two input, one output)
	malloc_test = cudaMalloc((void**)&dev_c, MATRIX_SIZE * sizeof(float));	// c
	if (malloc_test != cudaSuccess) printf("wtf1\t");

	malloc_test = cudaMalloc((void**)&dev_a, MATRIX_SIZE * sizeof(float));	// a
	if (malloc_test != cudaSuccess) printf("wtf2\t");

	malloc_test = cudaMalloc((void**)&dev_b, MATRIX_SIZE * sizeof(float));	// b
	if (malloc_test != cudaSuccess) printf("wtf3\t");

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_b, b, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);


	// initialize event-based timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	// Launch a kernel on the GPU with one thread for each element.
	dim3 threads = dim3(16, 16);
	dim3 blocks = dim3(MATRIX_DIM, MATRIX_DIM);// dim3(MATRIX_DIM / threads.x, MATRIX_DIM / threads.y);
	matrixAddition <<<blocks, threads>>>(dev_c, dev_a, dev_b);

	cudaError_t cErr = cudaGetLastError();
	if (cErr != cudaSuccess) printf("wtf\t");

	// addition finished, record time it took
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("time spent (ms): %.2f\n", gpu_time);

	//TODO
	// kernel with one thread per row

	// kernel with one thread per col


	cudaError_t syncErr = cudaDeviceSynchronize();
	if (syncErr != cudaSuccess) printf("wtf\t");

	// Copy output vector from GPU buffer to host memory.
	//printf("%f\n", dev_c[10]);
	cudaError_t cpyErr = cudaMemcpy(c, dev_c, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	
	/*for (int i = 0; i < MATRIX_SIZE; i++)
	{
		if (c[i] == 0) printf("%f, %f, %f\n", a[i], b[i], c[i]);
	}*/
	if (cpyErr != cudaSuccess) printf("wtf\t");

	if (false)
	{
		printf("testing printing of c");
		for (int i = 0; i < MATRIX_SIZE; i++)
			printf("%d\t\t%.2f\n", i, c[i]);
	}

	cudaError_t freeErr;
	freeErr = cudaFree(dev_c);
	if (freeErr != cudaSuccess) printf("wtf\t");	// c

	freeErr = cudaFree(dev_a);
	if (freeErr != cudaSuccess) printf("wtf\t");	// a

	freeErr = cudaFree(dev_b);
	if (freeErr != cudaSuccess) printf("wtf\t");	// b
}
