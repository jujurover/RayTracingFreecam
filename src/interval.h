#ifndef INTERVAL_H
#define INTERVAL_H

#include <cmath>
#include <limits>
#include <cstdlib>
#include "utility2.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct __align__(16) interval {
public:
    float min, max;

    __device__ __host__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __device__ __host__ interval(float min, float max) : min(min), max(max) {}

    __device__ __host__ float size() const {
        return max - min;
    }

    __device__ __host__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    __device__ __host__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    __device__ __host__ double clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __device__ __host__ interval expand(float delta) const {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    __device__ __host__ interval(const interval & a, const interval & b) {
        // Create the interval tightly enclosing the two input intervals.
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    static const interval empty, universe;
};

__device__ __host__ interval operator+(const interval& ival, float displacement) {
    return interval(ival.min + displacement, ival.max + displacement);
}

__device__ __host__ interval operator+(float displacement, const interval& ival) {
    return ival + displacement;
}

const interval interval::empty = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);


#endif