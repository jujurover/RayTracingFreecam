#ifndef QUADRILATERAL_H
#define QUADRILATERAL_H

#include "hittable.h"
#include "material.h"
#include "vec3_0.h"
#include "AABB.h"
#include "triangle.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


class __align__(16) quadrilateral : public hittable {
public:
    __device__ __host__ quadrilateral(const point3 & Q, const vec3 & u, const vec3 & v, Material * mat)
        : mat(mat), Q(Q), u(u), v(v)
    {
        // Create two triangles from the quadrilateral vertices.
        tri1 = triangle(Q, Q + u, Q + v, mat);
        tri2 = triangle(Q + u, Q + u + v, Q + v, mat);

        this->set_bounding_box();
        area = tri1.area + tri2.area;
    }

    __device__ __host__ void set_bounding_box() {
        AABB bbox_tri1 = tri1.bounding_box();
        AABB bbox_tri2 = tri2.bounding_box();
        bbox = AABB(bbox_tri1, bbox_tri2);
    }

    __device__ __host__ AABB bounding_box() const override { return bbox; }

    __device__ bool hit(const ray & r, interval ray_t, hit_record & rec) const override {
        if (tri1.hit(r, ray_t, rec) || tri2.hit(r, ray_t, rec))
        {
            float tex_u = dot(rec.p - Q, u) / u.length_squared();
            float tex_v = dot(rec.p - Q, v) / v.length_squared();
            rec.u = tex_u;
            rec.v = tex_v;
            return true;
        }
        return false;
    }

    __device__ vec3 get_uv(const point3 & p) const override {
        // Calculate the UV coordinates based on the point p
        float tex_u = dot(p - Q, u) / u.length_squared();
        float tex_v = dot(p - Q, v) / v.length_squared();
        return vec3(tex_u, tex_v, 0);
    }


    __device__ float pdf_value(const point3 & origin, const vec3 & direction) const override {
        hit_record rec;
        if (!hit(ray(origin, direction), interval(0.001, DBL_MAX), rec)) {
            return 0.0f;
        }

        float distance_squared = (origin - rec.p).length_squared();
        float dir_len = direction.length();

        if (dir_len < 1e-6f) return 0.0f; // avoid zero-length direction

        float cosine = fabs(dot(direction, rec.normal) / dir_len);

        if (cosine < 1e-6f || area <= 0.0f) {
            return 0.0f;
        }

        return distance_squared / (cosine * area);
    }


    __device__ vec3 random(const point3 & origin) const override {
        int rand_triangle = random_int(0, 1);
        vec3 p;
        if (rand_triangle == 0) {
            p = tri1.random(origin);
        }
        else {
            p = tri2.random(origin);
        }
        return p;
    }

    float area;
private:
    triangle tri1, tri2;
    Material* mat;
    AABB bbox;
    vec3 u, v, Q; // The two vectors defining the quadrilateral in 3D space.
};

#endif