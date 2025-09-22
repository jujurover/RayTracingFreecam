#ifndef UTILITY_H
#define UTILITY_H

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

__device__ const float infinity = FLT_MAX;
__device__ const float pi = 3.1415926535897932385;

inline __device__ __host__ float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

inline __device__ __host__ float random_double() {
#ifdef __CUDA_ARCH__
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int tId = (y * 1000 + x) * 10 + z;
    unsigned long long clk = clock(); // high-resolution GPU clock
    unsigned int h = tId * 374761393 + z * 668265263 + (unsigned int)(clk & 0xFFFFFFFF);
    h = (h ^ (h >> 13)) * 1274126177;
    h ^= h >> 16;
    return (h & 0xFFFFFF) / float(0x1000000); // float in [0,1)


#else
    return std::rand() / (RAND_MAX + 1.0f);
#endif
}



inline __device__ __host__ float random_double(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

inline __device__ __host__ int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return int(random_double(min, max + 1));
}


#endif