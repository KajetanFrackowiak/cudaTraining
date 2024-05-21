#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#define SIZE 256
#define SHMEM_SIZE 256 * sizeof(int)

__global__ void sum_reduction(int *v, int *v_r, clock_t *time) {
    if (threadIdx.x == 0) {
        time[blockIdx.x] = clock();
    }

    __shared__ int partial_sum[SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
        time[blockIdx.x + gridDim.x] = clock();
    }
}

void initialize_vector(int *h_v, int n) {
    for (int i = 0; i < n; ++i) {
        h_v[i] = rand() % 100;
    }
}

int main() {
    int N = 1 << 20;
    int bytes = N * sizeof(int);
    int *h_v, *h_v_r;
    int *d_v, *d_v_r;

    h_v = (int*)malloc(bytes);
    h_v_r = (int*)malloc(N / SIZE * sizeof(int));

    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, N / SIZE * sizeof(int));

    initialize_vector(h_v, N);

    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    int TB_SIZE = SIZE;
    int GRID_SIZE = N / TB_SIZE;

    // Allocate space for clock
    clock_t *time = (clock_t*)malloc(sizeof(clock_t) * GRID_SIZE * 2);
    clock_t *d_time;
    cudaMalloc(&d_time, sizeof(clock_t) * GRID_SIZE * 2);

    sum_reduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r, d_time);

    cudaMemcpy(h_v_r, d_v_r, GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(time, d_time, sizeof(clock_t) * GRID_SIZE * 2, cudaMemcpyDeviceToHost);

    std::ofstream outfile("output.dat");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }

    std::cout << "Block, Clocks" << std::endl;
    outfile << "Block, Clocks" << std::endl;
    for (int i = 0; i < GRID_SIZE; ++i) {
        std::cout << i << ",\t" << (time[i + GRID_SIZE] - time[i]) << std::endl;
        outfile << i << ",\t" << (time[i + GRID_SIZE] - time[i]) << std::endl;
    }

    outfile.close();

    free(h_v);
    free(h_v_r);
    free(time);
    cudaFree(d_v);
    cudaFree(d_v_r);
    cudaFree(d_time);

    return 0;
}
