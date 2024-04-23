#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>
#include <chrono>

void addCPU(int n, float* x, float* y) {
	for (int i = 0; i < n; i++) {
		y[i] = x[i] + y[i];
	}
}

__global__ void addGPU(int n, float* x, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index ; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}
}

int main() {
	int N = 1 << 20;

	float* xCPU = new float[N];
	float* yCPU= new float[N];

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		xCPU[i] = 1.0f;
		yCPU[i] = 2.0f;
	}	
	std::chrono::time_point<std::chrono::high_resolution_clock> startCPU, endCPU;
	startCPU = std::chrono::high_resolution_clock::now();
	addCPU(N, xCPU, yCPU);
	endCPU = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedCPU = endCPU - startCPU;
	float maxErrorCPU = 0.0f;

	for (int i = 0; i < N; i++) 
		maxErrorCPU = fmax(maxErrorCPU, fabs(yCPU[i] - 3.0f));

	std::cout << "CPU max error: " << maxErrorCPU << std::endl;
	std::cout << "CPU calculation time: " << elapsedCPU.count() * 1000 << " ms" << std::endl;

	delete[] xCPU;
	delete[] yCPU;

	float* xGPU, * yGPU;
	// Allocate Unified Memory - accessible from CPU or GPU
	cudaMallocManaged(&xGPU, N * sizeof(float));
	cudaMallocManaged(&yGPU, N * sizeof(float));

	for (int i = 0; i < N; i++) {
		xGPU[i] = 1.0f;
		yGPU[i] = 2.0f;
	}

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	std::chrono::time_point<std::chrono::high_resolution_clock> startGPU, endGPU;
	startGPU = std::chrono::high_resolution_clock::now();
	addGPU <<<numBlocks, blockSize>>> (N, xGPU, yGPU);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	endGPU = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedGPU = endGPU - startGPU;

	float maxErrorGPU = 0.0f;
	for (int i = 0; i < N; i++) 
		maxErrorGPU = fmax(maxErrorGPU, fabs(yGPU[i] - 3.0f));
	std::cout << "Max error in GPU calculations: " << maxErrorGPU << std::endl;
	std::cout << "GPU calculation time: " << elapsedGPU.count() * 1000 << " ms" << std::endl;

	cudaFree(xGPU);
	cudaFree(yGPU);

	return 0;
}