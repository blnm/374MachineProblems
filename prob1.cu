#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>

// necessary properties
// number and type of cuda devices
// clock rate
// num streaming multiprocessors (SM)
// num cores
// warp size
// amount of global mem
// amount of const mem
// amount of shared mem per block
// num of registers avail per block
// max num of threads per block
// max size of each dim of a block
// max size of each dim of a grid


int main()
{
	int nd;
	cudaGetDeviceCount(&nd);
	printf("number of cuda devices: %d\n", nd);
	for (int d = 0; d < nd; d++)
	{
		printf("device\t%d:\n", d);
		cudaDeviceProp dp;
		cudaGetDeviceProperties(&dp, d);

		// print relevant properties
		printf("\tdevice name:\t\t\t\t%s\n",				dp.name);
		printf("\tclock rate:\t\t\t\t%d\n",					dp.clockRate);
		printf("\tnumber of SMs:\t\t\t\t%d\n",				dp.multiProcessorCount);
		//printf("\tnumber of cores:\t%d\n", )						// need to calculate
		printf("\twarp size:\t\t\t\t%d\n",					dp.warpSize);
		printf("\tamount of global memory:\t\t%d\n",		dp.totalGlobalMem);
		printf("\tamount of const memory:\t\t\t%d\n",		dp.totalConstMem);
		printf("\tamount of shared memory per block:\t%d\n",dp.sharedMemPerBlock);
		printf("\tnumber of registers per block:\t\t%d\n",	dp.regsPerBlock);
		printf("\tmaximum threads per block:\t\t%d\n",		dp.maxThreadsPerBlock);
		printf("\tmaximum size of dim per block:\t\t%d\n",	dp.maxThreadsDim);
		printf("\tmaximul size of dim per grid:\t\t%d\n",	dp.maxGridSize);
		printf("\n");
	}
	
	
	return 0;
}