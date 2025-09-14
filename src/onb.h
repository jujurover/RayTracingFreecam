#ifndef ONB_H
#define ONB_H

#include "vec3_0.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct __align__(16) onb {
public:
    __device__ __host__ onb(const vec3 & n) {
        axis[2] = unit_vector(n);
        vec3 a = (axis[2].x() > 0.9 || -axis[2].x() > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        axis[1] = unit_vector(cross(axis[2], a));
        axis[0] = cross(axis[2], axis[1]);
    }

    __device__ __host__  const vec3& u() const { return axis[0]; }
    __device__ __host__ const vec3& v() const { return axis[1]; }
    __device__ __host__ const vec3& w() const { return axis[2]; }

    __device__ __host__ vec3 transform(const vec3 & v) const {
        // Transform from basis coordinates to local space.
        return (v[0] * axis[0]) + (v[1] * axis[1]) + (v[2] * axis[2]);
    }

private:
    vec3 axis[3];
};


#endif