#ifndef TEXTURE_H
#define TEXTURE_H
#include "color.h"
#include "vec3_0.h"
#include "hittable.h"

#include "rtw_stb_image.h"
//#include "perlin.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class texture {
public:
    __device__ __host__ virtual ~texture() = default;

    __device__ virtual color value(float u, float v, const hit_record& rec) const = 0;
};

class solid_color : public texture {
public:
    __device__ __host__ solid_color(const color& albedo) : albedo(albedo) {}

    __device__ __host__ solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {}

    __device__ color value(float u, float v, const hit_record& rec) const override {
        return albedo;
    }

private:
    color albedo;
};

class checker_texture : public texture {
public:
    __device__ __host__ checker_texture(float scale, texture* even, texture* odd)
        : inv_scale(1.0f / scale), even(even), odd(odd) {
    }

    __device__ __host__ checker_texture(float scale, const color& c1, const color& c2)
        : checker_texture(scale, new solid_color(c1), new solid_color(c2)) {
    }

    __device__ color value(float u, float v, const hit_record& rec) const override {
        int xInteger = int(floorf(inv_scale * rec.u));
        int yInteger = int(floorf(inv_scale * rec.v));

        bool isEven = (xInteger + yInteger) % 2 == 0;
        return isEven ? even->value(u, v, rec) : odd->value(u, v, rec);
    }

private:
    float inv_scale;
    texture* even;
    texture* odd;
};

class image_texture : public texture {
public:
    __device__ __host__ image_texture(rtw_image* img) : image(img) {}

    __device__ color value(float u, float v, const hit_record& rec) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image->height() <= 0) return color(0, 1, 1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = interval(0, 1).clamp(u);
        v = 1.0 - interval(0, 1).clamp(v);  // Flip V to image coordinates

        int i = int(u * image->width());
        int j = int(v * image->height());
        const unsigned char* pixel = image->pixel_data(i, j);
        float color_scale = 1.0f / 255.0f;
        
        return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }

private:
    rtw_image* image;
};


//class noise_texture : public texture {
//public:
//    noise_texture() {}
//    noise_texture(double scale, color texture_color) : scale(scale), texture_color(texture_color) {}
//
//
//    color value(double u, double v, const hit_record& rec) const override {
//        //return color(1, 1, 1) * 0.5 * (1.0 + noise.noise(scale * p));
//        //return color(1, 1, 1) * noise.turb(p, 7);
//        ///return color(.5, .5, .5) * (1 + std::sin(scale * rec.p_object_space.z() + 10 * noise.turb(rec.p_object_space, 7)));
//        return texture_color * noise.turb(rec.p, 7);
//
//    }
//
//private:
//    perlin noise;
//    double scale;
//    color texture_color;
//};
#endif