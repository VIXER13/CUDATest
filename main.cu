#include <iostream>
#include <omp.h>
#include "matrix_multiplication.cuh"

static constexpr size_t WARP_SIZE = 32;

int main() {
    try {
        const math::matrix<double> A{280 * WARP_SIZE, 280 * WARP_SIZE, 1.},
                                  B{280 * WARP_SIZE, 280 * WARP_SIZE, 2.};
        const double time = omp_get_wtime();
        const math::matrix<double> C = CUDA_test::matrix_multiplication(A, B);
        std::cout << "time = " << omp_get_wtime() - time << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "OK!" << std::endl;
    return EXIT_SUCCESS;
}