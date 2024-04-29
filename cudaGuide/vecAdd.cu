#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined (_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

__global__ void vecAdd(float* A, float* B, float* C, int numElements) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		C[i] = A[i] + B[i];
	}
}

int main(void) {
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	float* h_A, * h_B, * h_C;
	h_A = (float*)malloc(size);
	h_B = (float*)malloc(size);
	h_C = (float*)malloc(size);


	for (int i = 0; i < numElements; ++i) {
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	float* d_A, * d_B, * d_C;
	checkCuda(cudaMalloc((void**)&d_A, size));
	checkCuda(cudaMalloc((void**)&d_B, size));
	checkCuda(cudaMalloc((void**)&d_C, size));

	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	float ms; // elapsed time in miliseconds
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	checkCuda(cudaEventRecord(startEvent, 0));
	checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	vecAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);

	checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
	printf("Time for sequential transfer and execute (ms): %f\n", ms);

	checkCuda(cudaFree(d_A));
	checkCuda(cudaFree(d_B));
	checkCuda(cudaFree(d_C));
	free(h_A);
	free(h_B);
	free(h_C);
	
	return 0;
}