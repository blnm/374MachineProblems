#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MATRIX_DIM 255
#define MATRIX_SIZE (MATRIX_DIM * MATRIX_DIM)

void addWithCuda(float *c, const float *a, const float *b);

// assign
__global__ void matrixAddition(float *C, float *A, float *B)
{
	// calculate row, col index
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < MATRIX_DIM && col < MATRIX_DIM && row*MATRIX_DIM + col < MATRIX_SIZE)
	{
		float c_val = 0;
		int ind = row*MATRIX_DIM + col;
		
		C[ind] = A[ind] + B[ind];
		//printf("%f, %f, %f\n", A[ind], B[ind], C[ind]);
	}
	__syncthreads();
}

bool verifyGPUsoln(const float *GPU_C, const float *A, const float *B)
{
	float C[MATRIX_SIZE] = {};

	for (int i = 0; i < MATRIX_DIM; i++)
	{
		for (int j = 0; j < MATRIX_DIM; j++)
			C[i*MATRIX_DIM + j] = A[i*MATRIX_DIM + j] + B[i*MATRIX_DIM + j];
	}
	bool passed = true;
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		//printf("%f\t%f\n", GPU_C[i], C[i]);
		if (GPU_C[i] != C[i])
		{
			passed = false;
			//break;
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
	float a[MATRIX_SIZE] = {};
	float b[MATRIX_SIZE] = {};
    float c[MATRIX_SIZE] = {};

	for (int i = 0; i < MATRIX_SIZE; i++)
		a[i] = b[i] = (float)i;				//TODO - replace with rand()

	// Add vectors in parallel.
	printf("starting addition\n");
    addWithCuda(c, a, b);
	printf("addition finished\n");

	bool passed = verifyGPUsoln(c, a, b);
	//if (passed) printf("passed\n");
	//else printf("failed\n");


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
	dim3 threads = dim3(16, 16);// dim3(64, 1);
	dim3 blocks = dim3(MATRIX_SIZE / threads.x, MATRIX_SIZE / threads.y);
	matrixAddition <<<blocks, threads, 1, 0>>>(dev_c, dev_a, dev_b);
	
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
	cudaError_t cpyErr = cudaMemcpy(c, dev_c, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
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
