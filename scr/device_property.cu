#include "device_property.cuh"

#include <iostream>

#include <helper_cuda.h>
#include <helper_functions.h>

namespace CUDA {

void print_device_property() {
    int devices_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&devices_count));
    std::cout << "CUDA Devices count: " << devices_count << std::endl;
    cudaDeviceProp device_property = {};
    for(int dev = 0; dev < devices_count; ++dev) {
        checkCudaErrors(cudaGetDeviceProperties(&device_property, dev));
        std::cout << "CUDA Device            : " << dev << std::endl
                  << "Compute capability     : " << device_property.major << '.' << device_property.minor << std::endl
                  << "Name                   : " << device_property.name << std::endl
                  << "Total Global Memory    : " << device_property.totalGlobalMem << std::endl
                  << "Shared memory per block: " << device_property.sharedMemPerBlock << std::endl
                  << "Registers per block    : " << device_property.regsPerBlock << std::endl
                  << "Warp size              : " << device_property.warpSize << std::endl
                  << "Max threads per block  : " << device_property.maxThreadsPerBlock << std::endl
                  << "Total constant memory  : " << device_property.totalConstMem << std::endl
                  << "Clock Rate             : " << device_property.clockRate << std::endl
                  << "Texture Alignment      : " << device_property.textureAlignment << std::endl
                  << "Device Overlap         : " << device_property.deviceOverlap << std::endl
                  << "Multiprocessor Count   : " << device_property.multiProcessorCount << std::endl
                  << "Max Threads Dim        : " << device_property.maxThreadsDim[0] << ' '
                  << device_property.maxThreadsDim[1] << ' '
                  << device_property.maxThreadsDim[2] << std::endl
                  << "Max Grid Size          : " << device_property.maxGridSize[0] << ' '
                  << device_property.maxGridSize[1] << ' '
                  << device_property.maxGridSize[2] << std::endl;
    }
}

}