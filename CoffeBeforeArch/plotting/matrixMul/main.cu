#include <iostream>
#include <fstream>
#include "common.h"
#include <vector>

int main() {
    int N = 10;

    // Upper bound of matrix size
    int D = 1 << 10;

    std::vector<float> times;

    times = launch_mmul(D, N);

    std::ofstream output_file("tiled_mmul.dat", std::ios::out | std::ios::trunc);
    for (auto i : times) {
        output_file << i << "\n";
    }

    return 0;
}
