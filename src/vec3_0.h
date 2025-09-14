#pragma once

#include "utility2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct __align__(16) vec3 {
public:
    float e[3];

    __device__ __host__
        vec3() : e{ 0,0,0 } {}

    __device__ __host__
        vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

    __device__ __host__
        vec3(const vec3 & rhs) : e{ rhs.e[0], rhs.e[1], rhs.e[2] } {}

    __device__ __host__
        float x() const { return e[0]; }
    __device__ __host__
        float y() const { return e[1]; }
    __device__ __host__
        float z() const { return e[2]; }

    __device__ __host__
        vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __device__ __host__
        float operator[](int i) const { return e[i]; }
    __device__ __host__
        float& operator[](int i) { return e[i]; }

    __device__ __host__
        vec3& operator+=(const vec3 & v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __device__ __host__
        vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __device__ __host__
        vec3& operator/=(float t) {
        return *this *= 1 / t;
    }

    __device__ __host__
        float length() const {
        return sqrtf(length_squared());
    }

    __device__ __host__
        float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __device__ __host__
        bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
        return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s);
    }

    // These static functions are host-only, as random_double is not __device__.
    __device__ __host__ static vec3 random() {
        return vec3(random_double(), random_double(), random_double());
    }

    __device__ __host__ static vec3 random(float min, float max) {
        return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
    }

    __device__ __host__ void to_string(char* buffer, size_t buffer_size) const {
        // Format: "x y z"
        snprintf(buffer, buffer_size, "%g %g %g", e[0], e[1], e[2]);
    }

    __host__ __device__ vec3& operator=(const vec3 & r) = default;
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// CUDA-compatible vector utility functions
__device__ __host__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ __host__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ __host__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__device__ __host__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ __host__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__device__ __host__ inline vec3 operator/(const vec3& v, float t) {
    return (1 / t) * v;
}

__device__ __host__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__device__ __host__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ __host__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ __host__ inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (p.length_squared() < 1)
            return p;
    }
}

__device__ __host__ inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3::random(-1, 1);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1.0)
            return p / sqrtf(lensq);
    }
}

__device__ __host__ inline vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ __host__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ __host__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.0 - r_out_perp.length_squared())) * n;
    vec3 sum = r_out_perp + r_out_parallel;
    return r_out_perp + r_out_parallel;
}

__device__ __host__ inline vec3 lerp(const vec3& a, const vec3& b, float t) {
    return (1 - t) * a + t * b;
}

__device__ __host__ inline vec3 project(const vec3& v, const vec3& u) {
    return (dot(u, v) / (u.length_squared())) * u;
}

__device__ __host__ inline vec3 random_cosine_direction() {
    auto r1 = random_double();
    auto r2 = random_double();

    auto phi = 2 * pi * r1;
    auto x = cos(phi) * sqrt(r2);
    auto y = sin(phi) * sqrt(r2);
    auto z = sqrt(1 - r2);

    return vec3(x, y, z);
}
