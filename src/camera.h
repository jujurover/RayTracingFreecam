#ifndef CAMERA_H
#define CAMERA_H

#include "vec3_0.h"
#include "color.h"
#include "material.h"
#include "utility2.h"
#include <cfloat>
#include "background.h"
#include "pdf.h"
#include "BVHNode.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class __align__(16) camera_device {

public:

    int imageWidth;
    int imageHeight;
    int maxDepth;
    float verticalFov;
    vec3 lookFrom, lookAt, worldUpVector;
    float defocus_angle, focus_dist;
    float pixelSamplesScale;
    vec3 center, pixel00Coord, pixelDeltaU, pixelDeltaV, u, v, w;
    float upper_clamp;
    background* bg;

    __device__ camera_device()
        : imageWidth(800), imageHeight(600),
        maxDepth(50), verticalFov(40.0), lookFrom(point3(0, 0, 0)), lookAt(point3(0, 0, -1)),
        worldUpVector(vec3(0, 1, 0)), defocus_angle(0.0), focus_dist(1.0),
        pixelSamplesScale(1.0), upper_clamp(1.0)
    {
    }

    __device__ __host__ vec3 rotate(const vec3& axis, float angle, const vec3& v) {
        float c = cos(angle);
        float s = sin(angle);
        float t = 1 - c;
        vec3 a = unit_vector(axis);
        return vec3(v.x() * (t * a.x() * a.x() + c) + v.y() * (t * a.x() * a.y() - s * a.z()) + v.z() * (t * a.x() * a.z() + s * a.y()),
            v.x() * (t * a.x() * a.y() + s * a.z()) + v.y() * (t * a.y() * a.y() + c) + v.z() * (t * a.y() * a.z() - s * a.x()),
            v.x() * (t * a.x() * a.z() - s * a.y()) + v.y() * (t * a.y() * a.z() + s * a.x()) + v.z() * (t * a.z() * a.z() + c)
        );
    }

    __host__ void updatePitchYaw(float pitch, float yaw)
    {
        vec3 new_vec = rotate(worldUpVector, -yaw * 3.14159265f / 180.0f, -w);
        new_vec = rotate(cross(worldUpVector, lookFrom - new_vec), pitch * 3.14159265f / 180.0f, new_vec);
        lookAt = lookFrom + new_vec;
        initialize();
    }

    __device__ __host__ void initialize()
    {
        center = lookFrom;

        // Determine viewport dimensions.
        auto theta = verticalFov / 180 * 3.14159265;
        auto h = tan(theta / 2);
        auto viewportHeight = 2 * h * focus_dist;
        auto viewportWidth = viewportHeight * (double(imageWidth) / imageHeight);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookFrom - lookAt);
        u = unit_vector(cross(worldUpVector, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = float(viewportWidth) * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = float(viewportHeight) * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixelDeltaU = viewport_u / float(imageWidth);
        pixelDeltaV = viewport_v / float(imageHeight);

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00Coord = viewport_upper_left + 0.5f * (pixelDeltaU + pixelDeltaV);

        //determing clamping to reduce static
        /* if (samplesPerPixel > 2000)
            upper_clamp = 1000.0;
        else if (samplesPerPixel > 1000)
            upper_clamp = 100.0;
        else if (samplesPerPixel > 100)
            upper_clamp = 5.0;
        else
            upper_clamp = 1.0; */

    }

    __device__ color ray_color(const ray & r_in, int max_depth, const hittable & world, const hittable & lights, BVHNode & search_tree) {
        color final_color(0, 0, 0);     // accumulated color
        color attenuation(1.0, 1.0, 1.0); // keeps track of color scaling
        ray cur_ray = r_in;

        //initalizing variables here to avoid stack overflow
        pdf_record p_rec;
        interval MAX_interval(0.001, DBL_MAX);
        hit_record rec;
        color color_from_emission;
        ray scattered;
        color scatter_attenuation;
        color background;
        hittable_pdf lights_pdf(lights, rec.p);
        mixture_pdf mixed_pdf(lights_pdf, p_rec.get_pdf());

        float scattering_pdf;
        float pdf_val;


        for (int depth = max_depth; depth > 0; depth--) {



            if (!search_tree.hit(cur_ray, MAX_interval, rec)) {
                background = bg->getColor(unit_vector(r_in.direction()));
                final_color += attenuation * background;
                return final_color;
            }

            color_from_emission = rec.mat->emitted(rec, cur_ray);

            //russian roulette
            /*if (depth < maxDepth - 5) {
                float rr_prob = 0.5;
                if (random_double() > rr_prob)
                    return color_from_emission;
                attenuation /= rr_prob;
            }*/

            scattered;
            scatter_attenuation;

            if (!rec.mat->scatter(cur_ray, rec, scatter_attenuation, scattered, p_rec)) {
                // If material doesn�t scatter, only emission contributes
                final_color += attenuation * color_from_emission;
                return final_color;
            }

            //printf("p rec type = %d\n", p_rec.type);

            if (p_rec.type == p_rec.NONE) {
                // Specular reflection or refraction (deterministic)
                //printf("specular\n");
                attenuation = attenuation * scatter_attenuation;
                cur_ray = scattered;
                continue;
            }



            lights_pdf.set_origin(rec.p);
            mixed_pdf.set_pdfs(lights_pdf, p_rec.get_pdf());

            scattered = ray(rec.p, mixed_pdf.generate());

            pdf_val = mixed_pdf.value(scattered.direction()) + 1e-3;
            scattering_pdf = p_rec.value(scattered.direction());


            // accumulate emission
            final_color += (scattering_pdf * attenuation * color_from_emission) / pdf_val;

            // update attenuation & ray for next loop
            attenuation = (scattering_pdf * attenuation * scatter_attenuation) / pdf_val;
            cur_ray = scattered;

        }

        // If depth ran out, return what we gathered so far
        return clamp(final_color, 0.0, upper_clamp);
    }


    __device__ ray get_ray(int i, int j) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.

        auto pixel_sample = pixel00Coord
            + ((i + getOffsetAA()) * pixelDeltaU)
            + ((j + getOffsetAA()) * pixelDeltaV);



        auto ray_origin = center;
        auto ray_direction = unit_vector(pixel_sample - ray_origin);

        return ray(ray_origin, ray_direction);
    }



    __device__ double getOffsetAA() const
    {
        return random_double(-0.5, 0.5);
    }

    __device__ vec3 clamp(const vec3 & v, float min_val, float max_val) const {
        return vec3(
            fminf(fmaxf(v.x(), min_val), max_val),
            fminf(fmaxf(v.y(), min_val), max_val),
            fminf(fmaxf(v.z(), min_val), max_val)
        );
    }
};

#endif