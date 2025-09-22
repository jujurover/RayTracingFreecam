#ifndef COLOR_H
#define COLOR_H
using color = vec3;

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
namespace Colors
{
	const color WHITE(1.0, 1.0, 1.0);
	const color RED(1.0, 0.0, 0.0);
	const color GREEN(0.0, 1.0, 0.0);
	const color BLUE(0.0, 0.0, 1.0);
	const color BLACK(0.0, 0.0, 0.0);
	const color GREY(0.5, 0.5, 0.5);
	const color GRASS_GREEN(0.13, 0.55, 0.13);
	const color DARK_RED(0.5, 0.0, 0.0);
	const color DARK_GREEN(0.0, 0.5, 0.0);
	const color DARK_BLUE(0.0, 0.0, 0.5);
	const color SKY_BLUE(0.0, 0.709803922, 0.88627451);
}

#endif