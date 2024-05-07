<<<<<<< HEAD
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
=======
﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <assert.h>

>>>>>>> 4261020833e48df4be9de65c51a46c860151d785
#define MASK_LENGTH 7

__constant__ int mask[MASK_LENGTH];

<<<<<<< HEAD
    /*
    Args:
        array = padded array
        result = result array
        n = number of elements in array
    */

    __global__ void
    convolution_1d(int *array, int *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int mask_radius = MASK_LENGTH / 2;
    int start = tid - mask_radius;
    int temp = 0;

    for (int j = 0; j < MASK_LENGTH; ++j) {
        // Ignore elements that hang off (0s don't contribute)
        if (((start + j) >= 0) && (start + j < n)) {
            temp += array[start + j] * mask[j];
        }
    }
    result[tid] = temp;
}

void verify_result(int *array, int *mask, int *result, int n) {
    int mask_radius = MASK_LENGTH / 2;
    int temp;
    int start;
    for (int i = 0; i < n; ++i) {
        start = i - mask_radius;
        temp = 0;
        for (int j = 0; j < MASK_LENGTH; ++j) {
            if (((start + j) >= 0) && ((start + j) < n)) {
                temp += array[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}

int main() {
    int n = 1 << 20;
    int bytes_n = n * sizeof(int);
    // Size of the mask
    size_t bytes_m = MASK_LENGTH * sizeof(int);

    int *h_array = new int[n];
    for (int i = 0; i < n; ++i) {
        h_array[i] = rand() % 100;
    }

    int *h_mask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; ++i) {
        h_mask[i] = rand() % 10;
    }

    int *h_result = new int[n];

    int *d_array, *d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);

    // Copy the data directly to the symbol
    // Would require 2 API calls with cudaMemcpy
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;

    convolution_1d<<<GRID, THREADS>>>(d_array, d_result, n);

    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    verify_result(h_array, h_mask, h_result, n);

    std::cout << "COMPLETED SUCCESSFULLY!" << std::endl;

    delete[] h_array;
    delete[] h_mask;
    delete[] h_result;
    cudaFree(d_result);

    return 0;
=======
__global__ void convolution_1d(int* array, int* result, int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Radius of the mask
	int r = MASK_LENGTH / 2;

	int start = tid - r;
	int temp = 0;

	for (int j = 0; j < MASK_LENGTH; ++j) {
		if (((start + j) >= 0) && (start + j < n)) {
			temp += array[start + j] * mask[j];
		}
	}
	result[tid] = temp;
}

void verify_result(int* array, int* mask, int* result, int n) {
	int radius = MASK_LENGTH / 2;
	int temp;
	int start;
	for (int i = 0; i < n; ++i) {
		start = i - radius;
		temp = 0;
		for (int j = 0; j < MASK_LENGTH; ++j) {
			if ((start + j >= 0) && (start + j < n)) {
				temp += array[start + j] * mask[j]; }
		}
		assert(temp == result[i]);
	}
}

int main() {
	int n = 1 << 20;
	int bytes_n = n * sizeof(int);
	size_t bytes_m = MASK_LENGTH * sizeof(int);
	
	int* h_array = new int[n];
	for (int i = 0; i < n; ++i) {
		h_array[i] = rand() % 100;
	}

	int* h_mask = new int[MASK_LENGTH];
	for (int i = 0; i < MASK_LENGTH; ++i) {
		h_mask[i] = rand() % 10;
	}

	int* h_result = new int[n];

	int* d_array, * d_result;
	cudaMalloc(&d_array, bytes_n);
	cudaMalloc(&d_result, bytes_n);

	cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);

	// Copy the data directly to the symbol
	// Would require 2 API calls with cudaMemcpy
	cudaMemcpyToSymbol(mask, h_mask, bytes_m);

	int THREADS = 256;
	int GRID = (n + THREADS - 1) / THREADS;

	convolution_1d << <GRID, THREADS >> > (d_array, d_result, n);

	cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);
	
	verify_result(h_array, h_mask, h_result, n);
	
	std::cout << "COMPLETED SUCCESSFULLY!\n";

	delete[] h_array;
	delete[] h_result;
	delete[] h_mask;
	cudaFree(d_result);
	cudaFree(d_array);

	return 0;
>>>>>>> 4261020833e48df4be9de65c51a46c860151d785
}