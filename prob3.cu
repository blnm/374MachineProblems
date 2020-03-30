#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define MATRIX_DIM 500
#define MATRIX_SIZE (MATRIX_DIM * MATRIX_DIM)
#define BLOCK_WIDTH 16

// for selecting addition method. Selecting multiple at a time is fine,
#define ADD_BY_ELEMENT	true
#define ADD_BY_ROW		true
#define ADD_BY_COL		true
#define ADD_BY_CPU		true

void addWithCuda(float *p, const float *m, const float *n);
void testTransferTime(float *m, float *n);

// calculate one element with one thread
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
	}
	__syncthreads();
}

// calculate one row with one thread
__global__ void matrixAddByRow(float *C, float *A, float *B)
{
	// calculate col index
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if (col < MATRIX_DIM)
	{
		for (int row = 0; row < MATRIX_DIM; row++)
		{
			float c_val = 0;
			int ind = row*MATRIX_DIM + col;

			C[ind] = A[ind] + B[ind];
		}
	}
	__syncthreads();
}

// calculate one col with one thread
__global__ void matrixAddByCol(float *C, float *A, float *B)
{
	// calculate row index
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if (row < MATRIX_DIM)
	{
		for (int col = 0; col < MATRIX_DIM; col++)
		{
			float c_val = 0;
			int ind = row*MATRIX_DIM + col;

			C[ind] = A[ind] + B[ind];
		}
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
	float *c = (float *)malloc(MATRIX_SIZE * sizeof(float));

	for (int i = 0; i < MATRIX_SIZE; i++)
	{	// value between 0 and 10, one decimal place
		a[i] = rand() % 100 / 10.0;
		b[i] = rand() % 100 / 10.0;
	}

	testTransferTime(a, b);

	//addWithCuda(c, a, b);

	//verifyGPUsoln(c, a, b);

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
	malloc_test = cudaMalloc((void**)&dev_c, MATRIX_SIZE * sizeof(float));	// p
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_c\n");

	malloc_test = cudaMalloc((void**)&dev_a, MATRIX_SIZE * sizeof(float));	// m
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_a\n");

	malloc_test = cudaMalloc((void**)&dev_b, MATRIX_SIZE * sizeof(float));	// n
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_b\n");

	// Copy input vectors from host memory to GPU.
	cudaMemcpy(dev_a, a, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);


	// create event-based timers
	cudaEvent_t start, stop;
	float gpu_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// generall error for checking errors during GPU processing
	cudaError_t addErr;


	// create block/thread dims with add-by-element as default
	int numBlocks = MATRIX_DIM / BLOCK_WIDTH;
	if (MATRIX_DIM % BLOCK_WIDTH) numBlocks++;
	dim3 grid(numBlocks, numBlocks);
	dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

	if (ADD_BY_ELEMENT)
	{
		cudaEventRecord(start, 0); // start timer

		matrixAddition << </*blocks, threads*/grid, block >> >(dev_c, dev_a, dev_b);
		addErr = cudaGetLastError();
		if (addErr != cudaSuccess) printf("Error during addition: %s", cudaGetErrorString(addErr));


		cudaEventRecord(stop, 0);	// end timer and display results
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("one thread per element (ms):\t%.2f\n", gpu_time);
	}
	if (ADD_BY_ROW)
	{
		numBlocks = MATRIX_DIM / BLOCK_WIDTH;
		grid = dim3(numBlocks, numBlocks);
		block = dim3(BLOCK_WIDTH / 4, BLOCK_WIDTH / 4);

		cudaEventRecord(start, 0); // start timer

		matrixAddByRow << <grid, block >> >(dev_c, dev_a, dev_b);
		addErr = cudaGetLastError();
		if (addErr != cudaSuccess) printf("Error during addition: %s", cudaGetErrorString(addErr));

		cudaEventRecord(stop, 0);	// end timer and display results
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("one thread per row (ms):\t%.2f\n", gpu_time);
	}

	if (ADD_BY_COL)
	{
		// kernel with one thread per col
		numBlocks = MATRIX_DIM / BLOCK_WIDTH;
		grid = dim3(numBlocks, numBlocks);
		block = dim3(BLOCK_WIDTH / 4, BLOCK_WIDTH / 4);

		cudaEventRecord(start, 0); // start timer

		matrixAddByCol << <grid, block >> >(dev_c, dev_a, dev_b);
		addErr = cudaGetLastError();
		if (addErr != cudaSuccess) printf("Error during addition: %s", cudaGetErrorString(addErr));

		cudaEventRecord(stop, 0); // end timer and display results
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("one thread per col (ms):\t%.2f\n", gpu_time);
	}

	if (ADD_BY_CPU)
	{
		float cpu_time = 0;
		cudaEventRecord(start, 0); // start timer

								   // only care about operation time, so don't actually store results anywhere
		for (int i = 0; i < MATRIX_SIZE; i++)
			float temp = a[i] + b[i];

		cudaEventRecord(stop, 0); // end timer and display results
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&cpu_time, start, stop);
		printf("cpu time (ms):\t\t\t%.2f\n", cpu_time);
	}

	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaError_t cpyErr = cudaMemcpy(c, dev_c, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	if (cpyErr != cudaSuccess) printf("error copying dev_c to host\n");

	cudaError_t freeErr;
	freeErr = cudaFree(dev_c);
	if (freeErr != cudaSuccess) printf("error freeing dev_c\n");	// p

	freeErr = cudaFree(dev_a);
	if (freeErr != cudaSuccess) printf("error freeing dev_a\n");	// m

	freeErr = cudaFree(dev_b);
	if (freeErr != cudaSuccess) printf("error freeing dev_b\n");	// n
}


void testTransferTime(float *m, float *n)
{
	float *dev_m = 0;
	float *dev_n = 0;

	cudaError_t malloc_test;

	// Allocate GPU buffers for two vectors
	malloc_test = cudaMalloc((void**)&dev_m, MATRIX_SIZE * sizeof(float));	// m
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_a\n");

	malloc_test = cudaMalloc((void**)&dev_n, MATRIX_SIZE * sizeof(float));	// n
	if (malloc_test != cudaSuccess) printf("error allocating mem for dev_b\n");

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
	if (freeErr != cudaSuccess) printf("error freeing dev_c\n");	// m

	freeErr = cudaFree(dev_n);
	if (freeErr != cudaSuccess) printf("error freeing dev_a\n");	// n
}