#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

__global__ void matrixMul(int* a, int* b, int* c, int n) {
	// Compute each thread's row
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// Compute each thread's col
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int temp_sum = 0;
	// Boundary protection
	if ((row < n) && (col < n)) {
		// Iterate over row, and down column
		for (int k = 0; k < n; k++) {
			// Accumulate result for a single element
			temp_sum += a[row * n + k] * b[col * n + k]; // The transposed access of matrix b
		}
		// Assign result
		c[row * n + col] = temp_sum;
	}
}

// Initialization function for matrices
void matrix_init(int* a, int n) {
	for (int i = 0; i < n; i++) { // all the rows
		for (int j = 0; j < n; j++) { // all the columns
			a[i * n + j] = rand() % 100;
		}
	}
}

void check_answer(int* a, int* b, int* c, int n) {
	int* verify_c;
	verify_c = (int*)malloc(n * n * sizeof(int));
	int temp_sum;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			temp_sum = 0;
			for (int k = 0; k < n; k++) {
				temp_sum += a[i * n + k] * b[k * n + j];
			}
			verify_c[i * n + j] = temp_sum; // Store the computed value in verify_c
		}
	}

	// Check the result against the computed verify_c
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			assert(c[i * n + j] == verify_c[i * n + j]);
		}
	}
}

void transpose(int* a, int* a_t, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a_t[j * n + i] = a[i * n + j];
		}
	}
}

int main() {
	// Matrxi size 1024x1024
	int n = 1 << 10;

	size_t bytes = n * n * sizeof(int);

	int* h_a, * h_b, * h_c;
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	matrix_init(h_a, n);
	matrix_init(h_b, n);

	// Transpose matrix b
	int* h_b_t = (int*)malloc(bytes);
	transpose(h_b, h_b_t, n);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b_t, bytes, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(n + threadsPerBlock.y - 1) / threadsPerBlock.y);

	matrixMul << < numBlocks, threadsPerBlock >> > (d_a, d_b, d_c, n);

	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	check_answer(h_a, h_b, h_c, n);
	
	printf("COMPLETED SUCCESSFULLY\n");

	free(h_a);
	free(h_b);
	free(h_c);
	free(h_b_t);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
