#include "utils.h"



void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		getchar();
		if (abort) exit(code);
	}
}