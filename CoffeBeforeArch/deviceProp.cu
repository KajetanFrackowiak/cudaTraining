#include <cuda_runtime.h>

#include <iostream>

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "There are " << device_count << " GPU(s) in the system"
              << std::endl;

    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);

        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        std::cout << "Device " << i << " is a " << device_prop.name
                  << std::endl;

        int driver;
        int runtime;
        cudaDriverGetVersion(&driver);
        cudaRuntimeGetVersion(&runtime);

        std::cout << "CUDA capability: " << device_prop.major << "."
                  << device_prop.minor << std::endl;

        std::cout << "Global memory in GB: "
                  << device_prop.totalGlobalMem / (1 << 30) << std::endl;

        std::cout << "Number of SMs: " << device_prop.multiProcessorCount
                  << std::endl;

        std::cout << "Max clock rate: " << device_prop.clockRate * 1e-6 << "GHz"
                  << std::endl;

        std::cout << "The L2 cache size in MB: "
                  << device_prop.l2CacheSize / (1 << 20) << std::endl;

        std::cout << "Total shared memory per block in KB: "
                  << device_prop.sharedMemPerBlock / (1 << 10) << std::endl;
    }
}