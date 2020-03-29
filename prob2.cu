#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MATRIX_DIM 254
#define MATRIX_SIZE (MATRIX_DIM * MATRIX_DIM)

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
		//printf("%f, %f, %f\n", A[ind], B[ind], C[ind]);
	}
}

/*__global__ void matrixAdditionByCol(float *C, float *A, float *B)
{
	// calculate row index
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < MATRIX_DIM && col < MATRIX_DIM)
	{
		float c_val = 0;
		C[row*MATRIX_DIM + col] = A[row*MATRIX_DIM + col] + B[row*MATRIX_DIM + col];
	}
}*/

void verifyGPUsoln(const float *GPU_C, const float *A, const float *B)
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

	if (passed)
	{
		printf("TEST PASSED\n");
	}
	else 
		printf("TEST FAILED\n");
}


int main()
{
    //const int arraySize = 5;
	float a[MATRIX_SIZE] = {};// { 1, 2, 3, 4, 5 };
	float b[MATRIX_SIZE] = {};// { 10, 20, 30, 40, 50 };
    float c[MATRIX_SIZE] = {};

	for (int i = 0; i < MATRIX_SIZE; i++)
		a[i] = b[i] = (float)i;

	printf("starting addition\n");
    // Add vectors in parallel.
    addWithCuda(c, a, b);
	printf("addition finished\n");

	verifyGPUsoln(c, a, b);

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(float *c, const float *a, const float *b)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&dev_c, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_a, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_b, MATRIX_SIZE * sizeof(float));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_b, b, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
	dim3 threads = dim3(16, 16);// dim3(64, 1);
	dim3 blocks = dim3(MATRIX_SIZE / threads.x, MATRIX_SIZE / threads.y);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	//printf("%f, %f, %f", dev_a[0], dev_b[0], dev_c[0]);
	matrixAddition <<<blocks, threads, 1, 0>>>(dev_c, dev_a, dev_b);
	//printf("%f, %f, %f", dev_a[0], dev_b[0], dev_c[0]);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("time spent (ms): %.2f\n", gpu_time);
	//TODO
	// kernel with one thread per row

	// kernel with one thread per col

    
	cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
	cudaMemcpy(c, dev_c, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}
