#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction(int* v, int* v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Iterate of log base 2 the block dimension
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduce threads performing work by half previous the previous
		// Iteration each cycle
		if (threadIdx.x % (2 * s) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();

	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is indexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}

}

void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; ++i) {
		v[i] = 1;
	}
}

int main() {
	// Vector size
	int n = 1 << 16;
	size_t bytes = n * sizeof(int);

	int* h_v, * h_v_r;
	int* d_v, * d_v_r;

	// Allocate memory
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	initialize_vector(h_v, n);

	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	// TB size
	int TB_SIZE = SIZE;

	// Grid size
	int GRID_SIZE = (int)ceil(n / TB_SIZE);

	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);
	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	printf("Accumulated result is %d\n", h_v_r[0]);
	scanf("Press enter to continue: ");
	assert(h_v_r[0] == 65536);

	printf("COMPLETED SUCCESSFULLY!\n");

	cudaFree(d_v);
	cudaFree(d_v_r);
	free(h_v);
	free(h_v_r);

	return 0;
}
