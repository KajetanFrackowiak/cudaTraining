#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>

// #include <numeric>

// Number of bins for our plot
constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS);

// GPU kernel for computing a histogram
// Takes:
// a: Problem array in global memory
// result: result array
// N: Size of the array
__global__ void histogram(char* a, int* result, int N) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Calculate teh bin positions where threads are grouped together
	int alpha_position;
	for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
		// Calculate the position in the alphabet
		alpha_position = a[i] - 'a';
		atomicAdd(&result[alpha_position / DIV], 1);
	}
}

int main() {
	int N = 1 << 24;
	
	// Allocate memory on the host
	std::vector<char> h_input(N);
	// Allocate space for the binned result
	std::vector<int> h_result(BINS);

	// Initialize the array
	srand(1);
	std::generate(begin(h_input), end(h_input), []() { return 'a' + (rand() % 26); });

	// Allocate memory on the device
	char* d_input;
	int* d_result;
	cudaMalloc(&d_input, N);
	cudaMalloc(&d_result, BINS * sizeof(int));

	// Copy the array to the device
	cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int), cudaMemcpyDeviceToHost);

	int THREADS = 512;
	int BLOCKS = N / THREADS;

	histogram << <BLOCKS, THREADS >> > (d_input, d_result, N);

	cudaMemcpy(h_result.data(), d_result, BINS * sizeof(int), cudaMemcpyHostToDevice);

	assert(N == std::accumulate(h_result.begin(), h_result.end(), 0));

	// Write the data out for gnuplot
	std::ofstream output_file("histogram.dat", std::ios::out | std::ios::trunc);
	for (auto i : h_result) {
		output_file << i << "\n";
	}
	output_file.close();

	cudaFree(d_input);
	cudaFree(d_result);

	return 0;
}