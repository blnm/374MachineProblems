#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>

__global__ void increment_kernel(int *g_data, int inc_value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	g_data[idx] = g_data[idx] + inc_value;
}

int correct_output(int *data, const int n, const int x)
{
	for (int i = 0; i < n; i++)
		if (data[i] != x)
			return 0;
	return 1;
}

int main()
{
	cudaDeviceProp deviceProps;

	// get dev name
	cudaGetDeviceProperties(&deviceProps, 0);
	printf("CUDA device [%s]\n", deviceProps.name);

	int n = 16 * 1024 * 1024;
	int nbytes = n * sizeof(int);
	int value = 26;

	// alloc host mem
	int *a = 0;
	cudaMallocHost((void**)&a, nbytes);
	// memset(a, 0, nbytes); // commented out in source code, included just in case

	// alloc device mem
	int *d_a = 0;
	cudaMalloc((void**)&d_a, nbytes);
	cudaMemset(d_a, 255, nbytes);

	// set kernel launch config
	dim3 threads = dim3(512, 1);
	dim3 blocks = dim3(n / threads.x, 1);

	// create cuda event handles
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceSynchronize();
	float gpu_time = 0.0f;

	// asynchronously issue work to GPU (all to stream 0)
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);

	increment_kernel<<<blocks, threads, 0, 0 >>>(d_a, value);

	cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
	cudaEventRecord(stop, 0);

	// have CPU do some work while waiting for GPU to finish
	unsigned long int counter = 0;
	while (cudaEventQuery(stop) == cudaErrorNotReady)
	{
		counter++; // indicates that the CPU is running asynchronously while GPU is executing
	}

	cudaEventSynchronize(stop); // stop is updated here
	cudaEventElapsedTime(&gpu_time, start, stop); // time difference between start and stop

	// print the GPU times
	printf("time spent executing by the GPU: %.2f\n", gpu_time);
	printf("CPU executed %d iterations while waiting for GPU to finish\n", counter);

	// check the output for correctness
	bool bFinalResults = (bool)correct_output(a, n, value);

	// release resources
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFreeHost(a);
	cudaFree(d_a);
	cudaDeviceReset();

	return 0;
}