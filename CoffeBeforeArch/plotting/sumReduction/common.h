#include <vector>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#define LOWER_BOUND 256
#define BLOCK_DIM 256
#define SHMEM_SIZE 256 * 4

using namespace cooperative_groups;

/*
a: Pointer to the array
N: NUmber of array elements
*/
__global__ void sum_Reduction1(int* v, int* v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Increase the stride of the access until we exceed the CTA dimensions
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Change the indexing to be sequential threads
		int index = 2 * s * threadIdx.x;

		// Each thread does work unless the index goes off the block
		if (index < blockDim.x) {
			partial_sum[index] += partial_sum[index + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}
// -------------------------


// COOPERATIVE GROUPS

__device__ int reduce_sum(thread_group g, int* temp, int val) {
    int lane = g.thread_rank();

    // Each thread adds its partial sum[i] tu sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        temp[lane] = val;
        // wait for all threads to store
        g.sync();
        if (lane < i) {
            val += temp[lane + i];
        }
        // wait fo rall threads to load
        g.sync();
    }
    // only thread 0 will return full sum
    return val;
}

// Creates partials sums from the original array
__device__ int thread_sum(int* input, int n) {
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x) {
        // Cast as int4
        int4 in = ((int4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

// 
__global__ void sum_reduction2(int *sum, int *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Create partial sums from the array
    int my_sum = thread_sum(input, n);

    // Dynamic shared memory allocation
    extern __shared__ int temp[];

    // Identifier for a TB
    auto g = this_thread_block();

    // Reduce each TB
    int block_sum = reduce_sum(g, temp, my_sum);
    
    // Collect the partail result from each TB
    if (g.thread_rank() == 0) {
        atomicAdd(sum, block_sum);
    }
}
// -------------------------

// DEVICE FUNCTION
__device__ void warpReduce(volatile int* shmem_ptr, int t) {
    shmem_ptr[t] += shmem_ptr[t + 32];
    shmem_ptr[t] += shmem_ptr[t + 16];
    shmem_ptr[t] += shmem_ptr[t + 8];
    shmem_ptr[t] += shmem_ptr[t + 4];
    shmem_ptr[t] += shmem_ptr[t + 2];
    shmem_ptr[t] += shmem_ptr[t + 1];
}

__global__ void sum_reduction3(int *v, int* v_r) {
    // Allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements aAND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Store first partial result instead of just the elements
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    // Stop early (call device function instead)
    for (int s = blockDim.x / 2;  s > 32; s >>= 1) {
        //  Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is indexed by this block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}
// -------------------------
// DIVERGED

__global__ void sum_reduction4(int * v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    // Iterate of log base 2 the block dimension
    for (int s = 1; s < blockDim.x; s *= 2) {
        // Reduce the threads performing work by half previous the previous
        // iteration each cycle
        if (threadIdx.x % (2 * s) == 0) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0) {
        v_r[blockDim.x] = partial_sum[0];
    }
}
// -------------------------
// NoConflict

__global__ void sum_reduction5(int* v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];
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

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

// -------------------------
// ReduceIDLE
__global__ void sum_reduction6(int* v, int* v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];
    
    // Load elements AND do first add of reduction
    // vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Store first partial result instread of just elements
    partial_sum[threadIdx.x] = v[i] + v[i + blockIdx.x];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is furtehr than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Let the thread 0 for this block write it's result to main memory
    if (threadIdx.x == 0) {
        v_r[blockDim.x] = partial_sum[0];
    }
}

std::vector<float> launch_reduce(int D, int N) {
    int *h_a, *h_b;
    int *d_a, *d_b;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float exec_time;
    float total_time;

    std::vector<float> times;

    for (int i = LOWER_BOUND; i <= D; i += 128) {
        total_time;

        h_a = new int[i];
        h_b = new int[i];
        cudaMalloc(&d_a, i * sizeof(int));
        cudaMalloc(&d_b, i * sizeof(int));

        for (int j = 0; j < i; ++j) {
            h_a[j] = rand() % 100;
        }

        cudaMemcpy(d_a, h_a, i * sizeof(int), cudaMemcpyHostToDevice);
        
        dim3 block(BLOCK_DIM);
        dim3 grid((i + BLOCK_DIM - 1) / BLOCK_DIM);

        for(int j = 0; j < N; ++j) {
            cudaEventRecord(start);

            sum_reduction6<<<grid, block, BLOCK_DIM * sizeof(int)>>>(d_b, d_a);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&exec_time, start, stop);
            total_time += exec_time;            

            cudaMemcpy(h_b, d_b, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
        }
        times.push_back(total_time / N);

        delete[] h_a;
        delete[] h_b;
        cudaFree(d_a);
        cudaFree(d_b);
    }

    std::cout << "Completed Successfully" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return times;
}