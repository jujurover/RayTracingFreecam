#ifndef BVH_H
#define BVH_H

#include "AABB.h"
#include "hittable.h"
#include "hittableList.h"
#include "utility2.h"
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//shamelessly stolen from ray tracing in a week
class BVHNode {
public:
    __host__ BVHNode() : left(nullptr), right(nullptr), bbox() {}

    __host__ BVHNode(std::vector<hittable*>& objects, std::vector<int>& indices, std::vector<AABB> bboxes, size_t start, size_t end) {
        bbox = AABB::empty;
        for (size_t object_index = start; object_index < end; object_index++) {
            bbox = AABB(bbox, bboxes[object_index]);
        }

        size_t object_span = end - start;

        if (object_span == 1) {
            hittable_left = hittable_right = objects[indices[start]];
            is_leaf_node = true;
        }
        else if (object_span == 2) {
            hittable_left = objects[indices[start]];
            hittable_right = objects[indices[start + 1]];
            is_leaf_node = true;
        }
        else {
            // Use SAH to find the best split
            int best_axis = -1;
            size_t best_split = start;
            float best_cost = std::numeric_limits<float>::infinity();

            for (int axis = 0; axis < 3; axis++) {
                // Sort indices by the centroid of the bounding boxes along the current axis
                std::sort(indices.begin() + start, indices.begin() + end, [&](int a, int b) {
                    return bboxes[a].get_center()[axis] < bboxes[b].get_center()[axis];
                    });

                // Evaluate SAH for all possible splits
                for (size_t i = start + 1; i < end; i++) {
                    AABB left_bbox = AABB::empty;
                    AABB right_bbox = AABB::empty;

                    for (size_t j = start; j < i; j++) {
                        left_bbox = AABB(left_bbox, bboxes[indices[j]]);
                    }
                    for (size_t j = i; j < end; j++) {
                        right_bbox = AABB(right_bbox, bboxes[indices[j]]);
                    }

                    float SA_parent = bbox.surface_area();
                    float SA_left = left_bbox.surface_area();
                    float SA_right = right_bbox.surface_area();

                    size_t N_left = i - start;
                    size_t N_right = end - i;

                    float cost = 1.0f + (SA_left / SA_parent) * N_left + (SA_right / SA_parent) * N_right;

                    if (cost < best_cost) {
                        best_cost = cost;
                        best_axis = axis;
                        best_split = i;
                    }
                }
            }

            // Perform the best split
            if (best_axis != -1) {
                std::sort(indices.begin() + start, indices.begin() + end, [&](int a, int b) {
                    return bboxes[a].get_center()[best_axis] < bboxes[b].get_center()[best_axis];
                    });

                left = new BVHNode(objects, indices, bboxes, start, best_split);
                right = new BVHNode(objects, indices, bboxes, best_split, end);
            }
        }
    }

    __host__ void selection_sort(std::vector<AABB>& bboxes, std::vector<int>& indices, size_t start, size_t end, int axis) {
        for (size_t i = start; i < end - 1; i++) {
            size_t min_index = i;
            for (size_t j = i + 1; j < end; j++) {
                if (bboxes[indices[j]].axis_interval(axis).min < bboxes[indices[min_index]].axis_interval(axis).min) {
                    min_index = j;
                }
            }
            std::swap(indices[i], indices[min_index]);
            std::swap(bboxes[indices[i]], bboxes[indices[min_index]]);
        }
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        if (is_leaf())
        {
            bool hit_left = hittable_left->hit(r, ray_t, rec);
            bool hit_right = hittable_right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);
            return hit_left || hit_right;
        }

        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __device__ __host__ AABB bounding_box() const { return bbox; }

    __host__ __device__ bool is_leaf() const
    {
        return is_leaf_node;
    }

    __host__ void copy_to_device()
    {
        if (is_leaf_node) return;
        // Recursively copy child nodes to device memory.
        // first allocate memory for left and right nodes
        // then call the copy_to_device function on them
        // then set the left and right pointers to the device addresses
        BVHNode* d_left;
        BVHNode* d_right;
        cudaMallocManaged(&d_left, sizeof(BVHNode));
        cudaMallocManaged(&d_right, sizeof(BVHNode));
        left->copy_to_device();
        right->copy_to_device();
        cudaMemcpy(d_left, left, sizeof(BVHNode), cudaMemcpyHostToDevice);
        cudaMemcpy(d_right, right, sizeof(BVHNode), cudaMemcpyHostToDevice);
        delete left;
        delete right;
        left = d_left;
        right = d_right;
    }

private:
    BVHNode* left = nullptr;
    BVHNode* right = nullptr;
    hittable* hittable_left = nullptr;
    hittable* hittable_right = nullptr;
    AABB bbox;
    bool is_leaf_node = false;

    __host__ static bool box_compare(
        const hittable* a, const hittable* b, int axis_index
    ) {
        auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    __host__ static bool box_x_compare(const hittable* a, const hittable* b) {
        return box_compare(a, b, 0);
    }

    __host__ static bool box_y_compare(const hittable* a, const hittable* b) {
        return box_compare(a, b, 1);
    }

    __host__ static bool box_z_compare(const hittable* a, const hittable* b) {
        return box_compare(a, b, 2);
    }
};

#endif