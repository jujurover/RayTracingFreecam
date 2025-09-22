#pragma once

#include "vec3_0.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct ray {
    __device__ __host__
        ray() : orig(), dir() {}

    __device__ __host__
        ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    __device__ __host__
        const point3& origin() const { return orig; }

    __device__ __host__
        const vec3& direction() const { return dir; }

    __device__ __host__
        point3 at(float t) const {
        return orig + t * dir;
    }

    __host__ __device__ ray(const ray& r) = default;          // copy-ctor device-capable
    __host__ __device__ ray& operator=(const ray& r) = default;

private:
    point3 orig;
    vec3 dir;
};