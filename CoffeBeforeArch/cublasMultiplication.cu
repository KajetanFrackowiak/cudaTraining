#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

void verify_solution(float* a, float* b, float* c, int n) {
	float temp;
	float epsilon = 1e-3;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			temp = 0;
			for (int k = 0; k < n; ++k) {
				temp += a[k * n + i] * b[j * n + k];
			}
			assert(fabs(c[j * n + i] - temp) < epsilon);
		}
	}
}

int main() {
	int n = 1 << 10;
	size_t bytes = n * n * sizeof(float);

	float* h_a, * h_b, * h_c;
	float* d_a, * d_b, * d_c;

	h_a = (float*)malloc(bytes);
	h_b = (float*)malloc(bytes);
	h_c = (float*)malloc(bytes);
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Pseudo random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

	// Fill the matrix with random numbers on the device
	curandGenerateUniform(prng, d_a, n * n);

	// cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Scaling factors;
	float alpha = 1.0f;
	float beta = 0.0f;

	// Calcuale: c = (alpha * a) * b + (beta*c)
	// (m X n) * (n X k) = (m X k)
	// Signature: handle, operation, m, n, k, alpha, A, Ida, B ldb, beta, C ldc
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

	cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	verify_solution(h_a, h_b, h_c, n);

	printf("COMPLETED SUCCESSFULLY!\n");

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}