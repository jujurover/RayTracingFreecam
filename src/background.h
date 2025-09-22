#ifndef BACKGROUND_H
#define BACKGROUND_H
#include "color.h"
#include "vec3_0.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class background {
public:
	__device__ virtual color getColor(vec3 direction) const = 0;
	__device__ virtual color getIlluminationColor() const = 0;
};

class solid_background : public background {
public:
	__host__ __device__ solid_background(const color& c, const color& ic = color(1, 1, 1)) : bgColor(c), illuminationColor(ic) {}
	__device__ color getColor(vec3 direction) const override {
		return bgColor;
	}
	__device__ color getIlluminationColor() const override {
		return illuminationColor;
	}
private:
	color bgColor;
	color illuminationColor;
};

class gradient_background : public background {
public:
	__host__ __device__ gradient_background(const color& c1, const color& c2, const color& ic = color(1, 1, 1)) : color1(c1), color2(c2), illuminationColor(ic) {}
	__device__ color getColor(vec3 direction) const override {
		double t = 0.5 * (direction.y() + 1.0);
		return (1.0 - t) * color1 + t * color2;
	}
	__device__ color getIlluminationColor() const override {
		return illuminationColor;
	}

private:
	color color1;
	color color2;
	color illuminationColor;
};

//class image_background : public background {
//public:
//	image_background(const std::string& filename) {
//		// Load the image from the file
//		// This is a placeholder; actual image loading code should be implemented
//	}
//	virtual color getColor(vec3 direction) const override {
//		// Return the color from the image based on the direction
//		// This is a placeholder; actual image lookup code should be implemented
//		return color(1.0, 1.0, 1.0); // White color as a placeholder
//	}
//private:
//	//TODO
//};


#endif 
