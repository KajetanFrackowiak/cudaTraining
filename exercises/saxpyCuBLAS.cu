#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
	int N = 1 << 20;

	// Scalar multiplier
	float alpha = 2.0f; // multiplying by 2.0
	
	// Allocate memory on the CPU for host arrays x and y
	float* x = new float[N];
	float* y = new float[N];

	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	float* d_x;
	float* d_y;
	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	// Initialize cuBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	// Copy host arrays x and y to device arrays d_x and d_y
	cublasSetVector(N, sizeof(x[0]), x, 1, d_x, 1);
	cublasSetVector(N, sizeof(y[0]), y, 1, d_y, 1);

	// Perform SAXPY on 1M elements
	cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);

	// Copy result from device array d_y, to host array y
	cublasGetVector(N, sizeof(y[0]), d_y, 1, y, 1);

	delete[] x;
	delete[] y;
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);

	return 0;
}