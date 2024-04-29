#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <assert.h>
#include <cstdlib>
#include <numeric>

#define SHMEM_SIZE 256
__global__ void sumReduction(int* v, int* v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Increase the stride of the access until we exceed the CTA dimensions
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Change the indexing to be sequential threads
		int index = 2 * s * threadIdx.x;

		// Each thread does work unless the index goes off the block
		if (index < blockDim.x) {
			partial_sum[index] += partial_sum[index + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}
int main() {
	int N = 1 << 16;
	size_t bytes = N * sizeof(int);

	std::vector<int> h_v(N);
	std::vector<int> h_v_r(N);

	// initialize
	generate(begin(h_v), end(h_v), []() { return rand() % 10; });

	int* d_v, * d_v_r;
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);

	// TB size
	const int TB_SIZE = 256;

	// grid size
	int GRID_SIZE = N / TB_SIZE;

	sumReduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);
	sumReduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

	assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

	std::cout << "COMPLETED SUCCESSFULLY!" << std::endl;

	cudaFree(d_v);
	cudaFree(d_v_r);

	return 0;
}
