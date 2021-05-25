#include <iostream>
#include <omp.h>
#include <eigen3/Eigen/Dense>

#include "device_property.cuh"
#include "sum_vectors.cuh"

//#include "matrix_multiplication.cuh"

namespace {

template<class Vector>
void print_sum(const Vector& a, const Vector& b, const Vector& c, const size_t i) {
    std::cout << "index = " << i << " : " << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
}

}

int main() {
    try {
        CUDA::print_device_property();

        static constexpr size_t SIZE = 102400047;
        {   // CUDA
            const std::vector<float> a(SIZE, 5.f),
                                     b(SIZE, 3.f);
            const double time = omp_get_wtime();
            const std::vector<float> c = CUDA::sum_vectors(a, b);
            std::cout << "CUDA time = " << omp_get_wtime() - time << " s" << std::endl;
            print_sum(a, b, c, 5);
            print_sum(a, b, c, 5323);
            print_sum(a, b, c, SIZE-1);
        }

        std::cout << std::endl;

        {   // OMP
            const Eigen::VectorXf a = 5 * Eigen::VectorXf::Ones(SIZE),
                                  b = 3 * Eigen::VectorXf::Ones(SIZE);
            const double time = omp_get_wtime();
            const Eigen::VectorXf c = a + b;
            std::cout << "OMP time = " << omp_get_wtime() - time << " s" << std::endl;
            print_sum(a, b, c, 5);
            print_sum(a, b, c, 5323);
            print_sum(a, b, c, SIZE-1);
        }
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