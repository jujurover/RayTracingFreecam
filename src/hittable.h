#ifndef HITTABLE_H
#define HITTABLE_H
#include "ray.h"
#include "interval.h"
#include "vec3_0.h"
#include "AABB.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
struct Material;

struct hit_record {
public:
    point3 p;
    vec3 normal;
    Material* mat;
    float t;
    float u;
    float  v;
    point3 p_object_space;
};

struct bounds {
    point3 min;
    point3 max;
};


class hittable {
public:
    __device__ __host__ ~hittable() = default;

    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;

    __device__ __host__ virtual AABB bounding_box() const = 0;

    __device__ virtual float pdf_value(const point3& origin, const point3& sample_point) const {
        return 0.0;
    }

    __device__ virtual vec3 random(const point3& origin) const {
        return vec3(1, 0, 0);
    }

    __device__ virtual vec3 get_uv(const point3& p) const {
        // Default implementation returns (0, 0)
        return vec3(0, 0, 0);
    }
};


//class translate : public hittable {
//public:
//    translate(shared_ptr<hittable> object, const vec3& offset)
//        : object(object), offset(offset)
//    {
//        bbox = object->bounding_box() + offset;
//    }
//    
//    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
//        // Move the ray backwards by the offset
//        ray offset_r(r.origin() - offset, r.direction());
//
//        // Determine whether an intersection exists along the offset ray (and if so, where)
//        if (!object->hit(offset_r, ray_t, rec))
//            return false;
//
//        // Move the intersection point forwards by the offset
//        rec.p += offset;
//
//        return true;
//    }
//
//    AABB bounding_box() const override { return bbox; }
//
//
//private:
//    shared_ptr<hittable> object;
//    vec3 offset;
//    AABB bbox;
//};
//

struct  __align__(16) quat4 {
    float w, x, y, z;
    __device__ quat4(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}
    __device__ quat4() : w(0), x(0), y(0), z(0) {}
};

//rotations use tait-bryan euler angles
class __align__(16) rotator : public hittable {
public:
    //initalize rotator
    __device__ inline void initialize_bounding_box()
    {
        bbox = object->bounding_box();

        point3 min(FLT_MAX, FLT_MAX, FLT_MAX);
        point3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    float x = i * bbox.x.max + (1 - i) * bbox.x.min;
                    float y = j * bbox.y.max + (1 - j) * bbox.y.min;
                    float z = k * bbox.z.max + (1 - k) * bbox.z.min;

                    //translate xyz with respect to the origin
                    point3 vertex = point3(x, y, z) - bbox.get_center();
                    quat4 vertex_q;
                    point_to_quarternion(vertex, vertex_q);
                    quat4 rotated_vertex_q;
                    active_rotation(vertex_q, q, rotated_vertex_q);
                    point3 rotated_vertex;
                    quarternion_to_point(rotated_vertex_q, rotated_vertex);

                    for (int c = 0; c < 3; c++) {
                        min[c] = fminf(min[c], rotated_vertex[c]);
                        max[c] = fmaxf(max[c], rotated_vertex[c]);
                    }
                }
            }
        }

        bbox = AABB(min + bbox.get_center(), max + bbox.get_center());
        bbox.pad_to_minimums();
    }


    __device__ rotator(hittable * object, float x_rotate, float y_rotate, float z_rotate) : object(object) {
        //convert rotations to quarternion
        float angle_x = degrees_to_radians(x_rotate);
        float angle_y = degrees_to_radians(y_rotate);
        float angle_z = degrees_to_radians(z_rotate);

        float q0 = cosf(angle_x / 2) * cosf(angle_y / 2) * cosf(angle_z / 2) + sinf(angle_x / 2) * sinf(angle_y / 2) * sinf(angle_z / 2);
        float q1 = sinf(angle_x / 2) * cosf(angle_y / 2) * cosf(angle_z / 2) - cosf(angle_x / 2) * sinf(angle_y / 2) * sinf(angle_z / 2);
        float q2 = cosf(angle_x / 2) * sinf(angle_y / 2) * cosf(angle_z / 2) + sinf(angle_x / 2) * cosf(angle_y / 2) * sinf(angle_z / 2);
        float q3 = cosf(angle_x / 2) * cosf(angle_y / 2) * sinf(angle_z / 2) - sinf(angle_x / 2) * sinf(angle_y / 2) * cosf(angle_z / 2);

        q = quat4(q0, q1, q2, q3);


        initialize_bounding_box();
    }

    __device__ __forceinline__ inline void point_to_quarternion(const point3 & p, quat4 & out) const
    {
        // Quaternion representation: [w, x, y, z]
        // For a point/vector, w = 0
        out = quat4(0.0, p.x(), p.y(), p.z());
    }

    __device__ __forceinline__ inline void quarternion_to_point(const quat4 & q, point3 & out) const
    {
        // Assumes q = [w, x, y, z], and w should be 0 for a point/vector quaternion
        out = point3(q.x, q.y, q.z);
    }

    __device__ __forceinline__ inline void multiply(const quat4 & q1, const quat4 & q2, quat4 & out)const
    {
        out = quat4(q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w - q1.y * q2.z + q1.z * q2.y,
            q1.w * q2.y + q1.x * q2.z + q1.y * q2.w - q1.z * q2.x,
            q1.w * q2.z - q1.x * q2.y + q1.y * q2.x + q1.z * q2.w);
    }

    __device__ __forceinline__ inline void conjugate(const quat4 & q, quat4 & out) const {
        out = quat4(q.w, -q.x, -q.y, -q.z);
    }

    __device__ __forceinline__ inline void active_rotation(const quat4 & p, const quat4 & q, quat4 & out) const {
        quat4 tmp, tmp2;
        conjugate(q, tmp);
        multiply(p, q, tmp2);
        multiply(tmp, tmp2, out);
    }



    __device__ __forceinline__ bool hit(const ray & r, interval ray_t, hit_record & rec) const override {

        point3 origin = r.origin();
        point3 center = bbox.get_center();

        quat4 rotator_q = q;
        quat4 q_conj;
        conjugate(rotator_q, q_conj);


        // converting origin to object space
        quat4 tmp;
        point_to_quarternion(origin - center, tmp);
        active_rotation(tmp, q_conj, tmp);
        point3 untranslated_transformed_origin;
        quarternion_to_point(tmp, untranslated_transformed_origin);
        untranslated_transformed_origin += center;

        // converting direction to object space
        point_to_quarternion(r.direction(), tmp);
        active_rotation(tmp, q_conj, tmp);
        vec3 transformed_direction;
        quarternion_to_point(tmp, transformed_direction);

        //initiate transformed ray in object space
        ray rotated_r = ray(untranslated_transformed_origin, transformed_direction);

        if (!object->hit(rotated_r, ray_t, rec)) {
            return false;
        }

        //store the intersection point in object space
        rec.p_object_space = rec.p;

        // Transform the intersection point from object space back to world space.
        point_to_quarternion(rec.p - center, tmp);
        active_rotation(tmp, rotator_q, tmp);
        quarternion_to_point(tmp, rec.p);
        rec.p += center;

        // Transform the normal vector from object space back to world space.
        point_to_quarternion(rec.normal, tmp);
        active_rotation(tmp, rotator_q, tmp);
        quarternion_to_point(tmp, rec.normal);
        rec.normal = unit_vector(rec.normal);

        return true;
    }

    __device__ __host__ __forceinline__ AABB bounding_box() const override { return bbox; }

    __device__ float pdf_value(const point3 & origin, const point3 & sample_point) const override {
        quat4 conj_q;
        conjugate(q, conj_q);
        point3 center = bbox.get_center();


        quat4 tmp;
        point_to_quarternion(origin - center, tmp);
        active_rotation(tmp, conj_q, tmp);
        point3 untranslated_transformed_origin;
        quarternion_to_point(tmp, untranslated_transformed_origin);
        untranslated_transformed_origin += center;

        point_to_quarternion(sample_point - center, tmp);
        active_rotation(tmp, conj_q, tmp);
        point3 untranslated_transformed_sample;
        quarternion_to_point(tmp, untranslated_transformed_sample);
        untranslated_transformed_sample += center;

        float pdf = object->pdf_value(untranslated_transformed_origin, untranslated_transformed_sample);

        return pdf;
    }


    __device__ vec3 random(const point3 & origin) const override {
        quat4 conj_q;
        conjugate(q, conj_q);
        point3 center = bbox.get_center();

        quat4 tmp;
        point_to_quarternion(origin - center, tmp);
        active_rotation(tmp, conj_q, tmp);
        point3 origin_transformed;
        quarternion_to_point(tmp, origin_transformed);
        origin_transformed += center;

        // Rotate sampled direction back to world space
        point_to_quarternion(object->random(origin_transformed), tmp);
        active_rotation(tmp, q, tmp);
        vec3 dir_world;
        quarternion_to_point(tmp, dir_world);

        return unit_vector(dir_world);
    }

private:
    hittable* object;
    quat4 q;
    AABB bbox;
};


#endif