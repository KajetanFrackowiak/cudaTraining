#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Static shmem calculation for convenience (int 16x16 matrix)
#define SHMEM_SIZE 16 * 16 * 4

__global__ void titledMatrixMul(int *a, int *b, int *c, int n, int tile_size) {
    // Two statistically-sized pieces of shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // Shorten these parameters for clean re-use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate global row and column positions for this thread
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // Intermediate sum for element being written
    int temp_val = 0;

    // Sweep tiles over entire matrix
    for (int i = 0; i < (n / tile_size); i++) {
        /*
            Every thread in a threadb,ock loads one element into shared memory.
            The elemetn location in shared memory corresponds to the thread's 
            position in the threadblock (e.g. thread[0, 0] loads for
            A[0 * tile_size + 0], and B[0 * tile_size + 0],)

             Explanation of indexing parameters
             For A:
                row*n: Indexes the global row for this thread (loop-invariant)
                i*tile_size: Indexes the new set of colunns each iteration
                tx: Indexes the column within that set
            For B:
                i*tile_size*n: INdexes the next set of rows each iteration
                ty*n: Indexes the row within that set
                col: Indexes the global column (loop_invariant)
        */
       B
    }
}