#ifndef CUDATEST_SUM_VECTORS_CUH
#define CUDATEST_SUM_VECTORS_CUH

#include <stdexcept>
#include <string>
#include <vector>
#include <cinttypes>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

namespace CUDA {

template<class T>
__global__ void sum_vectors_kernel(T *const res, const T *const lhs, const T *const rhs, const size_t size) {
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        res[idx] = lhs[idx] + rhs[idx];
}

template<size_t Threads = 64, class T>
std::vector<T> sum_vectors(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    if (lhs.size() != rhs.size())
        throw std::logic_error{"lhs.size() != rhs.size(); lhs.size() = " + std::to_string(lhs.size()) + "; rhs.size() = " + std::to_string(rhs.size()) + ';'};

    T* d_lhs = nullptr,
     * d_rhs = nullptr,
     * d_res = nullptr;
    const size_t bytes_count = lhs.size() * sizeof(T);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_lhs), bytes_count));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_rhs), bytes_count));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_res), bytes_count));

    cudaStream_t stream = nullptr;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaMemcpyAsync(d_lhs, lhs.data(), bytes_count, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_rhs, rhs.data(), bytes_count, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    const auto div = std::imaxdiv(lhs.size(), Threads);
    sum_vectors_kernel<<<dim3(div.quot + (div.rem != 0)), dim3(Threads), 0, stream>>>(d_res, d_lhs, d_rhs, rhs.size());
    std::vector<T> res(rhs.size());
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaMemcpyAsync(res.data(), d_res, bytes_count, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(d_lhs));
    checkCudaErrors(cudaFree(d_rhs));
    checkCudaErrors(cudaFree(d_res));

    return std::move(res);
}

}

#endif