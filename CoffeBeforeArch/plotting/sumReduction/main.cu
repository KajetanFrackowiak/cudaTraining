#include <iostream>
#include <fstream>
#include <vector>
#include "common.h"

int main() {
    // Number of iterations to run per-kernel
    int N = 10;

    // Upper bound of array size
    int D = 1 << 14;

    std::vector<float> times;
    // Get execution time for naive implementation
    times = launch_reduce(D, N);

    // Write out the times to a data file
    std::ofstream output_file("reduce_idle.dat", std::ios::out | std::ios::trunc);
    for (auto i : times) {
        output_file << i << "\n";
    }

    return 0;
}