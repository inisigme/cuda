#include "tests.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	c[i] = a[i] + b[i];
}