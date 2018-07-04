#include "cuda_includes.h"

__global__ void addKernel(int *c, const int *a, const int *b);