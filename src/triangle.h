#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.h"
#include "material.h"
#include "vec3_0.h"
#include <array>

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class __align__(16) triangle : public hittable {
public:
    __device__ __host__ triangle() {}
    __device__ __host__ triangle(const point3 & P1, const point3 & P2, const point3 & P3, Material * mat) : p1(P1), p2(P2), p3(P3), material(mat)
    {
        this->set_bounding_box();
        normal = unit_vector(cross(p2 - p1, p3 - p1));
        area = 0.5 * cross(p2 - p1, p3 - p1).length();
    }

    float area;


    /* triangle(const point3& Q, const vec3& u, const vec3& v, shared_ptr<Material> mat)
     {
         triangle(Q, Q + u, Q + v, mat);
     }*/


    __device__ __host__ void set_bounding_box() {
        // Compute the bounding box of all vertices.
        vec3 side1 = p2 - p1;
        vec3 side2 = p3 - p1;
        vec3 side3 = p3 - p2;


        interval intervals[3];

        for (int axis_idx = 0; axis_idx < 3; axis_idx++) // Fixed loop index range
        {
            vec3 axis = vec3(
                (axis_idx == 0) ? 1 : 0,
                (axis_idx == 1) ? 1 : 0,
                (axis_idx == 2) ? 1 : 0
            );


            //project each side onto the axis
            vec3 proj1 = project(side1, axis);
            vec3 proj2 = project(side2, axis);
            vec3 proj3 = project(side3, axis);

            //find the min and max of the projections
            float a = p1[axis_idx] + proj1[axis_idx];
            float b = p1[axis_idx] + proj2[axis_idx];
            float c = p2[axis_idx] + proj3[axis_idx];

            float min_proj = fminf(fminf(a, b), c);
            float max_proj = fmaxf(fmaxf(a, b), c);

            //set the bounding box intervals
            intervals[axis_idx] = interval(min_proj, max_proj); // Fixed indexing
        }

        bbox = AABB(intervals[0], intervals[1], intervals[2]);
    }

    __device__ __host__ AABB bounding_box() const override { return bbox; }

    __device__ bool hit(const ray & r, interval ray_t, hit_record & rec) const override {
        point3 S = r.origin();
        vec3 d = r.direction();
        vec3 N = normal;

        if (dot(N, d) > 0.0f) N = -N;   //ensure ray faces surface

        if (fabs(dot(N, d)) < 0.00001f) return false; //if parallel

        float t = (dot(N, p1) - dot(S, N)) / dot(d, N); //determining intersection point
        if (t < 0.0005f) return false; //if intersection is behind the ray origin, return no hit

        vec3 P = S + d * t; //intersection point calculation

        //determinine if the intersection point is inside the triangle
        //check if cross products of edges and vector to intersection point are all in the same direction
        vec3 c1 = cross(p2 - p1, P - p1);
        vec3 c2 = cross(p3 - p2, P - p2);
        vec3 c3 = cross(p1 - p3, P - p3);
        vec3 n = normal;
        if (dot(c1, n) < 0 || dot(c2, n) < 0 || dot(c3, n) < 0) return false;

        if (!ray_t.surrounds(t)) return false;

        rec.t = t;
        rec.p = r.at(rec.t);
        rec.normal = normal;
        rec.mat = material;

        return true;
    }

    __device__ float pdf_value(const point3 & origin, const vec3 & direction) const override {
        hit_record rec;
        if (!this->hit(ray(origin, direction), interval(0.001, DBL_MAX), rec))
            return 0;

        auto distance_squared = rec.t * rec.t * direction.length_squared();
        auto cosine = fabs(dot(direction, rec.normal) / direction.length());

        return distance_squared / (cosine * area);
    }

    __device__ vec3 random(const point3 & origin) const override {
        float r1 = random_double();
        float r2 = random_double();

        point3 sample = (1 - sqrt(r1)) * p1 + (sqrt(r1) * (1 - r2)) * p2 + (sqrt(r1) * r2) * p3;
        return unit_vector(sample - origin);
    }

private:
    point3 p1, p2, p3;
    vec3 normal;
    Material* material;
    AABB bbox;
};

#endif