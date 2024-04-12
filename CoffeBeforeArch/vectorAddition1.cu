#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// CUDA kernel for vector addiction
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    // Calculate global thead ID (tid)
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Vector boundary guard
    if (tid < n) {
        // Each thread adds a single element
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Vector size of 2^16 (65536 elements)
    int n = 2 << 16;
    // Host vector pointers
    int *h_a, *h_b, *h_c;
    // Device vector pointer
    int *d_a, *d_b, *d_c;
    // Allocation size for all vectors
    size_t bytes = sizeof(int) * n;

    // Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize vectors a and b with random values between 0 and 99
    matrix_init(h_a, n);
    matrix_init(h_b, n);

    // Copy data from
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threadblock size
    int NUM_THREADS = 256;

    // Grid size
    int NUM_BLOCK = (int)ceil(n / NUM_THREADS);

    // Launch kernel on default stream w/o shmem
    vectorAdd<<<NUM_BLOCK, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // Check result for errors
    error_check(h_a, h_b, h_c, n);
    
}