#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>

// Number of bins for our bins
constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS);

/*
a: Problem array in global memory
result: result array
N: Size of the array
*/
__global__ void histogram(char* a, int* result, int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate a local histogram for each TB
	__shared__ int s_result[BINS];

	// Allocate the shared memory to 0
	if (threadIdx.x < BINS) {
		s_result[threadIdx.x] = 0;
	}

	// Wait for shared memory writes to complete
	__syncthreads();

	// Calculate the bin positions locally
	int alpha_position;
	for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
		// Calculate the position in the alphabet
		alpha_position = a[i] - 'a';
		atomicAdd(&s_result[(alpha_position / DIV)], 1);
	}

	// Wait for shared memory writes to complete
	__syncthreads;

	// Combine the partail results
	if (threadIdx.x < BINS) {
		atomicAdd(&result[threadIdx.x], s_result[threadIdx.x]);
	}
}

int main() {
	// Declare our problem size
	int N = 1 << 16;

	std::vector<char> h_input(N);
	std::vector<int> h_result(BINS);

	srand(1);
	std::generate(begin(h_input), end(h_input), []() { return 'a' + (rand() % 26); });

	char* d_input;
	int* d_result;
	cudaMalloc(&d_input, N);
	cudaMalloc(&d_result, BINS * sizeof(int));

	cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int), cudaMemcpyHostToDevice);

	int THREADS = 512;
	int BLOCKS = N / THREADS;

	histogram << <BLOCKS, THREADS >> > (d_input, d_result, N);

	cudaMemcpy(h_result.data(), d_result, BINS * sizeof(int), cudaMemcpyDeviceToHost);
	
	assert(N == std::accumulate(begin(h_result), end(h_result), 0));

	// Dump the counts of the bins to a file
	std::ofstream output_file("shmem_atomic.bat", std::ios::out | std::ios::trunc);
	for (auto i : h_result) {
		output_file << i << "\n";
	}
	output_file.close();

	cudaFree(d_result);
	cudaFree(d_input);

	return 0;
}