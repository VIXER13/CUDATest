#ifndef CUDATEST_MATRIX_MULTIPLICATION_CUH
#define CUDATEST_MATRIX_MULTIPLICATION_CUH

#include "matrix.hpp"
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

namespace CUDA_test {

template<class T, size_t Warp_Size = 32>
__global__ void matrix_multiplication_CUDA(T *const C, const T *const A, const T *const B, const size_t wA, const size_t wB) {
    const size_t aBegin = wA * Warp_Size * blockIdx.y, // Index of the first sub-matrix of A processed by the block
                 aEnd   = aBegin + wA - 1;              // Index of the last sub-matrix of A processed by the block
    T Csub = 0; // Csub is used to store the element of the block sub-matrix that is computed by the thread
    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for(size_t a = aBegin, b = blockIdx.x * Warp_Size; a <= aEnd; a += Warp_Size, b += wB * Warp_Size) {
        __shared__ T As[Warp_Size][Warp_Size], // Declaration of the shared memory array As used to store the sub-matrix of A
                     Bs[Warp_Size][Warp_Size]; // Declaration of the shared memory array Bs used to store the sub-matrix of B
        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        As[threadIdx.y][threadIdx.x] = A[a + wA * threadIdx.y + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[b + wB * threadIdx.y + threadIdx.x];
        __syncthreads(); // Synchronize to make sure the matrices are loaded
#pragma unroll // Multiply the two matrices together; each thread computes one element of the block sub-matrix
        for(size_t k = 0; k < Warp_Size; ++k)
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads(); // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
    }
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    const size_t c = wB * Warp_Size * blockIdx.y + Warp_Size * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

template<class T, size_t Warp_Size = 32>
math::matrix<T> matrix_multiplication(const math::matrix<T>& A, const math::matrix<T>& B) {
    if (A.cols() != B.rows())
        throw std::logic_error{"matrix_multiplication: A.cols() != B.rows(); A.cols() = " + std::to_string(A.cols()) + ", B.rows() = " + std::to_string(B.rows())};

    math::matrix<T> C{A.rows(), B.cols()};
    T* d_A = nullptr,
     * d_B = nullptr,
     * d_C = nullptr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), A.size() * sizeof(T)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), B.size() * sizeof(T)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), C.size() * sizeof(T)));

    cudaStream_t stream = nullptr;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaMemcpyAsync(d_A, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    dim3 threads{Warp_Size, Warp_Size},
         grid(B.cols() / threads.x, A.rows() / threads.y);
    matrix_multiplication_CUDA<T, Warp_Size><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, A.rows(), B.rows());

    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaMemcpyAsync(C.data(), d_C, C.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    return std::move(C);
}

}

#endif //CUDATEST_MATRIX_MULTIPLICATION_CUH