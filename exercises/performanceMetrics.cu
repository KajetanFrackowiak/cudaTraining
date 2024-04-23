#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__
void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(void)
{
    int N = 20 * (1 << 20);
    float* x, * y, * d_x, * d_y;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    // Perform SAXPY on 1M elements
    saxpy << <(N + 511) / 512, 512 >> > (N, 2.0f, d_x, d_y);

    cudaEventRecord(stop);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 4.0f));
    }

    printf("Max error: %f\n", maxError);
    // N*4 is the number of bytes trasferred per array read or write
    // and the factor of three represents the readif of x and the reading of writing of y
    printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6); 
    

    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}