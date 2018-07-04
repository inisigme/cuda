#include "stdafx.h"
#include "cuda_includes.h"
#include "utils.h"
#include "tests.cuh"
#include "time.h"

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);

	gpuErrchk(cudaMalloc((void**)&dev_c, size * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_a, size * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_b, size * sizeof(int)));


	gpuErrchk(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));


	addKernel <<<1, size >>> (dev_c, dev_a, dev_b);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	return cudaStatus;
}


int main(int argc, char *argv[])
{
	try {
		const unsigned int size = 1000;
		std::vector<int> a(size);
		std::vector<int> b(size);
		std::vector<int> c(size,0);

		for (auto i : a) i = rand();
		for (auto i : b) i = rand();
		

		gpuErrchk(addWithCuda(&c[0], &a[0], &b[0], size));


		gpuErrchk(cudaDeviceReset());

		std::cout << "end" << std::endl;
		getchar();
		return 0;
	}
	catch (std::exception& exc) {
		cerr << exc.what() << std::endl;
		getchar();
		exit(EXIT_FAILURE);
	}
	return (EXIT_SUCCESS);
}