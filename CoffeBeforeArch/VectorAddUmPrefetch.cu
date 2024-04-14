#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cassert>
#include <iostream>

__global__ void vectorAdd(int* a, int* b, int* c, int N) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Boundary check
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

int main() {
	const int N = 1 << 16; // 2^16
	size_t bytes = N * sizeof(int);

	// Declare unified memory pointers
	int* a, * b, * c;

	// Allocation memory for these pointers
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	// Get the device ID for prefetching calls
	int id = cudaGetDevice(&id);

	// Set some hints about the data and some prefetching
	cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	cudaMemPrefetchAsync(c, bytes, id);

	// Initialize vectors
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	// Pre-fetch 'a' and 'b' arrays to the specified device (GPU)
	cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
	cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);

	// Threads per CTA (1024 threads per CTA)
	int BLOCK_SIZE = 1 << 10;

	// CTAs per Grid
	int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Call CUDA kernel
	vectorAdd << <GRID_SIZE, BLOCK_SIZE >> > (a, b, c, N);

	// Wait for all previous operations before using values
	// We need this because we don't get the implicit synchronization of
	// cudaMemcoy like tin the original example
	cudaDeviceSynchronize();

	// Prefetch to the host (CPU)
	cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
	cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	// Verify the result on the CPU
	for (int i = 0; i < N; i++) {
		assert(c[i] == a[i] + b[i]);
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	std::cout << "COMPLETED SUCCESSFULLY!\n";

	return 0;
}