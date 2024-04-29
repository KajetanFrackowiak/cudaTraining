#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <cassert>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction(int* v, int* v_r) {
	__shared__ int partial_sum[SHMEM_SIZE];

	// load elements AND do first add of reduction
	// vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockIdx.x];
	__syncthreads();

	// start at  1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
	
	// let the thread 0 for this block write it's result to main memory
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; ++i) {
		v[i] = 1; // rand() % 10
	}
}

int main() {
	int n = 1 << 16; // 65536
	size_t bytes = n * sizeof(int);

	int* h_v, * h_v_r;
	int* d_v, * d_v_r;

	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);
	
	initialize_vector(h_v, n);

	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	int TB_SIZE = SIZE;
	int GRID_SIZE = n / TB_SIZE / 2;

	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r); // computes the sum reduction on the input vector
	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r); // reduces the result from the first kernel call

	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	// Print the result
	printf("Accumulated result is %d\n", h_v_r[0]);
	assert(h_v_r[0] = 1 << 16);

	printf("COMPLETED SUCCESSFULLY!\n");

	return 0;
}