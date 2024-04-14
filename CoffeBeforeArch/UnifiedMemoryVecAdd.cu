﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

__global__ void VectorAddUM(int* a, int* b, int* c, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Boundary check
	if (tid < n) {
		c[tid] = a[tid] + b[tid];
	}
}

void init_vector(int* a, int* b, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}
}

void check_answer(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {
	// Get the device ID for other CUDA ceils
	int id = cudaGetDevice(&id);

	// Declare number of elements per array
	int n = 1 << 16;

	// Size of each arrays in bytes
	size_t bytes = n * sizeof(int);

	// Declare unified memory pointers
	int* a, *b, *c;

	// Allocation memory for these pointers
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	// Initialize vectors
	init_vector(a, b, n);

	// Set threadBlock size
	int BLOCK_SIZE = 256;

	// Set grid size
	int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Cell CUDA kernel
	// Uncomment these for pre-fetching 'a' and 'b' vectors to device
	// cudaMemoryPrefetchAsync(a, bytes, id);
	// cudaMemoryPrefetchAsync(b, bytes, id);
	VectorAddUM << <GRID_SIZE, BLOCK_SIZE >> > (a, b, c, n);

	// Wait for all previous operations before using values
	cudaDeviceSynchronize();

	// Uncomment this for pre-fetching 'c' to the host
	 //cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	// Check result
	check_answer(a, b, c, n);

	printf("COMPLETED SUCCESSFULLY");
	
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}
