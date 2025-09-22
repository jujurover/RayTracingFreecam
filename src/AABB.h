#ifndef AABB_H
#define AABB_H

#include "interval.h"
#include "vec3_0.h"
#include "ray.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct __align__(16) AABB {
public:
    interval x, y, z;

    __device__ __host__ AABB() {} //empty box


    //box with bounds
    __device__ __host__ AABB(const interval & x, const interval & y, const interval & z)
        : x(x), y(y), z(z)
    {
        pad_to_minimums();
    }


    __device__ __host__ AABB(const point3 & a, const point3 & b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.

        x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
        y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
        z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
        pad_to_minimums();

    }

    __device__ __host__ AABB(const AABB & box0, const AABB & box1) {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
    }


    __device__ __host__ const interval& axis_interval(int n) const {
        if (n == 0) return x;
        if (n == 1) return y;
        if (n == 2) return z;
    }

    __device__ __host__ bool hit(const ray & r, interval ray_t) const {
        const point3& ray_orig = r.origin();
        const vec3& ray_dir = r.direction();

        for (int axis = 0; axis < 3; axis++) {
            const interval& ax = axis_interval(axis);
            const double adinv = 1.0 / ray_dir[axis];

            float t0 = (ax.min - ray_orig[axis]) * adinv;
            float t1 = (ax.max - ray_orig[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            }
            else {
                if (t1 > ray_t.min) ray_t.min = t1;
                if (t0 < ray_t.max) ray_t.max = t0;
            }

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }

    __device__ __host__ int longest_axis() const {
        // Returns the index of the longest axis of the bounding box.

        if (x.size() > y.size())
            return x.size() > z.size() ? 0 : 2;
        else
            return y.size() > z.size() ? 1 : 2;
    }

    __device__ __host__ void pad_to_minimums() {
        // Adjust the AABB so that no side is narrower than some delta, padding if necessary.

        float delta = 0.0001f;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }

    __device__ __host__ point3 get_center() const {
        // Returns the center point of the bounding box.
        return point3((x.max + x.min) / 2, (y.max + y.min) / 2, (z.max + z.min) / 2);
    }

    __device__ __host__ float surface_area() const {
        // Returns the surface area of the bounding box.
        float dx = x.size();
        float dy = y.size();
        float dz = z.size();
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

    static const AABB empty, universe;
};

__device__ __host__ AABB operator+(const AABB& bbox, const vec3& offset) {
    return AABB(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

__device__ __host__ AABB operator+(const vec3& offset, const AABB& bbox) {
    return bbox + offset;
}

const AABB AABB::empty = AABB(interval::empty, interval::empty, interval::empty);
const AABB AABB::universe = AABB(interval::universe, interval::universe, interval::universe);

#endif