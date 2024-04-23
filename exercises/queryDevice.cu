#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>

__global__ void saxpy(int n, float a, float* x, float* y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
	int N = 20 * (1 << 20);
	float* x, * y, * d_x, * d_y;
	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));
	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start);
	saxpy << <(N + 255) / 256, 256 >> > (N, 2.0, d_x, d_y);
	cudaEventRecord(stop);

	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);
	printf("Calculation time: %fms\n", miliseconds);

	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) {
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	}
	if (errAsync != cudaSuccess) {
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
		//printf("Async kernel error: %s\n", cudaGetErrorString(cudaGetLastError()))
	}



	int nDevices;

	cudaError_t err = cudaGetDeviceCount(&nDevices);

	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
	for (int i = 0; i < nDevices; ++i) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("\tDevice name: %s\n", prop.name);
		printf("\tMemory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("\tPeak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}

	free(x);
	free(y);
	cudaFree(d_x);
	cudaFree(d_y);

	return 0;
}