#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "material.h"
#include "vec3_0.h"
#include <memory>
#include "onb.h"

#include "AABB.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class __align__(16) sphere : public hittable {
public:
    __device__ __host__
        sphere(point3 center, float radius, Material * mat)
        : center(center), radius((radius > 0) ? radius : 0.0), mat(mat)
    {
        auto rvec = vec3(radius, radius, radius);
        bbox = AABB(center - rvec, center + rvec);
    }

    __device__ bool hit(const ray & r, interval ray_t, hit_record & rec) const override {
        vec3 oc = r.origin() - center;

        float a = dot(r.direction(), r.direction());
        float half_b = dot(oc, r.direction());     // note: not multiplied by 2
        float c = dot(oc, oc) - radius * radius;

        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;

        float sqrtd = sqrt(discriminant);

        // Find nearest root in range
        float root = (-half_b - sqrtd) / a;


        if (!ray_t.surrounds(root)) {
            root = (-half_b + sqrtd) / a;
            if (!ray_t.surrounds(root)) return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        rec.p_object_space = rec.p; 

        rec.mat = mat;
        get_sphere_uv(rec.normal, rec.u, rec.v);
        return true;
    }

    __device__ __host__ AABB bounding_box() const override { return bbox; }

    __device__ static void get_sphere_uv(const point3 & p, float& u, float& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2 * pi);
        v = theta / pi;
    }

    __device__ float pdf_value(const point3 & origin, const vec3 & direction) const override {
        hit_record rec;
        if (!this->hit(ray(origin, direction), interval(0.001, infinity), rec))
            return 0;

        auto dist_squared = (center - origin).length_squared();
        auto cos_theta_max = sqrt(1 - radius * radius / dist_squared);
        auto solid_angle = 2 * pi * (1 - cos_theta_max);

        return  1 / solid_angle;
    }

    __device__ vec3 random(const point3 & origin) const override {
        vec3 direction = center - origin;
        auto distance_squared = direction.length_squared();
        onb uvw(direction);
        return uvw.transform(random_to_sphere(radius, distance_squared));
    }


private:
    point3 center;
    float radius;
    Material* mat;
    AABB bbox;

    __device__ static vec3 random_to_sphere(float radius, float distance_squared) {
        auto r1 = random_double();
        auto r2 = random_double();
        auto z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

        auto phi = 2 * pi * r1;
        auto x = cos(phi) * sqrt(1 - z * z);
        auto y = sin(phi) * sqrt(1 - z * z);

        return vec3(x, y, z);
    }
};


#endif