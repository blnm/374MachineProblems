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

// gets number of cores based on major/minor revision number
// return of -1 indicates unknown device.
// sources for number of cores per SM: 
//	https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
//	https://devblogs.nvidia.com/inside-pascal/
int getNumCores(cudaDeviceProp devProp)
{
	int mp = devProp.multiProcessorCount;

	switch (devProp.major)
	{
	// excluding revision 1.x as CUDA 8 does not support these cards.
	case 2: // Fermi
		return (devProp.minor == 1) ? (mp * 48) : (mp * 32);
	case 3: // Kepler
		return mp * 192;
	case 5: // Maxwell
		return mp * 128;
	case 6: // Pascal
		if (devProp.minor == 1 || devProp.minor == 2) return mp * 128;
		else if (devProp.minor == 0) return mp * 64;
		else return -1;
	case 7: // Volta, Turing
		if (devProp.minor == 0 || devProp.minor == 5) return mp * 64;
		else return -1;
	default:
		return -1;
	}
}

int main()
{
	int num_devs;
	cudaGetDeviceCount(&num_devs);
	printf("number of cuda devices: %d\n", num_devs);
	for (int dev = 0; dev < num_devs; dev++)
	{
		printf("device\t%d:\n", dev);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, dev);

		// print relevant properties
		printf("\tdevice name:\t\t\t\t%s\n",			devProp.name);
		printf("\tclock rate:\t\t\t\t%d\n",			devProp.clockRate);
		printf("\tnumber of SMs:\t\t\t\t%d\n",			devProp.multiProcessorCount);
		printf("\tnumber of cores:\t\t\t%d\n",			getNumCores(devProp));
		printf("\twarp size:\t\t\t\t%d\n",			devProp.warpSize);
		printf("\tamount of global memory:\t\t%d\n",		devProp.totalGlobalMem);
		printf("\tamount of const memory:\t\t\t%d\n",		devProp.totalConstMem);
		printf("\tamount of shared memory per block:\t%d\n",	devProp.sharedMemPerBlock);
		printf("\tnumber of registers per block:\t\t%d\n",	devProp.regsPerBlock);
		printf("\tmaximum threads per block:\t\t%d\n",		devProp.maxThreadsPerBlock);
		printf("\tmaximum size of dim per block:\t\t%d\n",	devProp.maxThreadsDim);
		printf("\tmaximul size of dim per grid:\t\t%d\n",	devProp.maxGridSize);
		printf("\n");
	}
	
	
	return 0;
}
