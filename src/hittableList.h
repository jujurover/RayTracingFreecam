#ifndef HITTABLELIST_H
#define HITTABLELIST_H
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
class __align__(16) hittable_list : public hittable {
public:
    hittable** objects;   // points to preallocated device/UM array
    int capacity;
    int num_objects;

    __host__ __device__ hittable_list() : objects(nullptr), capacity(0), num_objects(0) {}

    __device__ void add(hittable * obj) {
        if (num_objects < capacity) {
            objects[num_objects++] = obj;
            bbox = AABB(bbox, obj->bounding_box());
        }
        else
        {
            printf("error: too many objects added");

        }
    }

    __device__ bool hit(const ray & r, interval ray_t, hit_record & rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < num_objects; i++) {
            hittable* obj = *(objects + i);  // pointer arithmetic instead of objects[i]

            if (obj->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    __device__ float pdf_value(const point3 & origin, const vec3 & direction) const override {
        //if (num_objects == 0) return 1.0;

        auto weight = 1.0 / num_objects;
        auto sum = 0.0;

        double number;

        for (int i = 0; i < num_objects; i++) {
            hittable* obj = *(objects + i);  // pointer arithmetic instead of objects[i]
            number = obj->pdf_value(origin, direction);
            sum += weight * number;
        }

        return sum;
    }

    __device__ vec3 random(const point3 & origin) const override {
        int idx = random_int(0, num_objects - 1);
        hittable* obj = *(objects + idx);  // pointer arithmetic instead of objects[i]
        return obj->random(origin);
    }

    __host__ __device__ AABB bounding_box() const override { return bbox; }

private:
    AABB bbox;
};
#endif