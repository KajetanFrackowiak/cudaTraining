#ifndef COMMON_H
#define COMMON_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SHMEM_SIZE 256 * 4

constexpr int LOWER_BOUND = 128; // Define LOWER_BOUND as an example

/*
m: Pointer to the matrix
N: Dimension of the matrix
*/
void init_matrix(int* m, int N) {
    for (int i = 0; i < N * N; ++i) {
        m[i] = rand() % 100;
    }
}

/*
a: Pointer to input matrix 'a'
b: Pointer to input matrix 'b'
c: Pointer to output matrix 'c'
N: Dimension of the matrix
*/
__global__ void naive_mmul(int* a, int* b, int* c, int N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int temp = 0;

    for (int i = 0; i < N; ++i) {
        temp += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = temp;
}

__global__ void aligned_mmul(int* a, int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    for (int i = 0; i < n; ++i) {
        temp_sum += a[i * n + row] * b[i * n + col];
    }
    c[row * n + col] = temp_sum;
}

__global__ void tiled_mmul(int* a, int* b, int* c, int N) {
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int temp_val = 0;

    // Sweep tiles over entire matrix
    for (int i = 0; i < (N / blockDim.x); ++i) {
        // Load sum_matrices into shared memory
        A[(ty * blockDim.x) + tx] = a[row * N + (i * blockDim.x + tx)];
        B[(ty * blockDim.x) + tx] = b[(i * blockDim.x * N + ty * N) + col];

        // Ensure all threads have loaded their data before proceeding
        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j) {
            temp_val += A[(ty * blockDim.x) + j] * B[(j * blockDim.x) + tx];
        }

        __syncthreads();
    }
    c[row * N + col] = temp_val;
}

std::vector<float> launch_mmul(int D, int N) {
    int BLOCK_DIM = 16;
    int GRID_DIM;
    int* h_a, * h_b, * h_c;
    int* d_a, * d_b, * d_c;

    // Start and stop event times
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Variables to collect timing information
    float exec_time;
    float total_time;

    std::vector<float> times;

    for (int i = LOWER_BOUND; i <= D; i += 128) {
        // Re-initialize total_time each iteration
        total_time = 0;

        h_a = new int[i * i];
        h_b = new int[i * i];
        h_c = new int[i * i];
        cudaMalloc(&d_a, i * i * sizeof(int));
        cudaMalloc(&d_b, i * i * sizeof(int));
        cudaMalloc(&d_c, i * i * sizeof(int));

        // Initialize the input matrices
        init_matrix(h_a, i);
        init_matrix(h_b, i);
        cudaMemcpy(d_a, h_a, i * i * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, i * i * sizeof(int), cudaMemcpyHostToDevice);

        // Calculate grid dimension and create launch parameters
        GRID_DIM = i / BLOCK_DIM;
        dim3 grid(GRID_DIM, GRID_DIM);
        dim3 block(BLOCK_DIM, BLOCK_DIM);

        // Average execution time for "N" kernel runs
        for (int j = 0; j < N; ++j) {
            cudaEventRecord(start);
            // Uncomment which implementation you would like to profile
            // naive_mmul << <grid, block >> > (d_a, d_b, d_c, i);
            // aligned_mmul<<<grid, block>>>(d_a, d_b, d_c, i);
            tiled_mmul<<<grid, block>>>(d_a, d_b, d_c, i);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&exec_time, start, stop);
            total_time += exec_time;

            cudaMemcpy(h_c, d_c, i * i * sizeof(int), cudaMemcpyDeviceToHost);
        }
        // Add the average time to the vector
        times.push_back(total_time / N);

        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    std::cout << "Completed Successfully!" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return times; // Return the times vector
}

#endif // COMMON_H
