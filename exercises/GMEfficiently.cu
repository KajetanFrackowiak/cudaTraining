﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cstring>

// Convience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

template <typename T>
__global__ void offset(T* a, int s) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i] = a[i] + 1;
}

template <typename T>
__global__ void stride(T* a, int s) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
	a[i] = a[i] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB) {
	int blockSize = 256;
	float ms;

	T* d_a;
	cudaEvent_t startEvent, stopEvent;

	int n = nMB * 1024 * 1024 / sizeof(T);

	// NB: d_a(33*nMB) for stride case
	checkCuda(cudaMalloc(&d_a, n * 33 * sizeof(T)));

	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	printf("Offset, bandwidth (GB/s):\n");

	offset << <n / blockSize, blockSize >> > (d_a, 0); // warm up

	for (int i = 0; i <= 32; ++i) {
		checkCuda(cudaMemset(d_a, 0, n * sizeof(T)));

		checkCuda(cudaEventRecord(startEvent, 0));
		offset << <n / blockSize, blockSize >> > (d_a, i);
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));

		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		printf("%d, %f\n", i, 2 * nMB / ms);
	}

	printf("\n");
	printf("Stride, Bandwidth (GB/s):\n");

	stride << <n / blockSize, blockSize >> > (d_a, 1); // warm up
	for (int i = 1; i <= 32; ++i) {
		checkCuda(cudaMemset(d_a, 0, n * sizeof(T)));

		checkCuda(cudaEventRecord(startEvent, 0));
		stride << <n / blockSize, blockSize >> > (d_a, i);
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));

		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		printf("%d, %f\n", i, 2 * nMB / ms);

		}
		checkCuda(cudaEventDestroy(startEvent));
		checkCuda(cudaEventDestroy(stopEvent));
		cudaFree(d_a);
}

int main(int argc, char** argv) {
	int nMB = 4;
	int deviceId = 0;
	bool bFp64 = false;

	for (int i = 1; i < argc; ++i) {
		if (!strncmp(argv[i], "dev=", 4))
			deviceId = atoi((char*)(&argv[i][4]));
		else if (!strcmp(argv[i], "fp64"))
			bFp64 = true;
	}

	cudaDeviceProp prop;

	checkCuda(cudaSetDevice(deviceId));
	checkCuda(cudaGetDeviceProperties(&prop, deviceId));
	printf("Device: %s\n", prop.name);
	printf("Transfer size (MB): %d\n", nMB);

	printf("%s Precision", bFp64 ? "Double" : "Single");

	if (bFp64) runTest<double>(deviceId, nMB);
	else runTest<float>(deviceId, nMB);

}