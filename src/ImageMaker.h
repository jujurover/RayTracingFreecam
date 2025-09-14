#ifndef IMAGE_MAKER_H
#define IMAGE_MAKER_H

#include <iostream>
#include <filesystem>
#include <cmath>
#include "./svpng.inc"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
namespace ImageMaker {
	__host__ float clamp(double value, double min, double max) {
		return fmax(min, fmin(max, value));
	}

	__host__ void imshow(double* SRC, int WIDTH, int HEIGHT)
	{
		unsigned char* image = new unsigned char[WIDTH * HEIGHT * 3];
		unsigned char* p = image;
		double* S = SRC;


		FILE* fp;
		errno_t err = fopen_s(&fp, "image.png", "wb");
		if (err != 0 || fp == nullptr) {
			std::cerr << "Failed to open file for writing.\n";
			delete[] image;
			return;
		}

		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{




				*p++ = (unsigned char)(clamp(pow(*S, 1.0 / 2.2) * 255, 0.0, 255.0)); S++;
				*p++ = (unsigned char)(clamp(pow(*S, 1.0 / 2.2) * 255, 0.0, 255.0)); S++;
				*p++ = (unsigned char)(clamp(pow(*S, 1.0 / 2.2) * 255, 0.0, 255.0)); S++;

				//print color values
				//printf("Color: %d %d %d\n", p[-3], p[-2], p[-1]);



			}
		}



		svpng(fp, WIDTH, HEIGHT, image, 0);
		fclose(fp); // Ensure the file is closed
		// Print the saved image path without using std::filesystem
		char fullpath[FILENAME_MAX];
		if (_fullpath(fullpath, "image.png", FILENAME_MAX)) {
			std::cout << "Saved image to: " << fullpath << std::endl;
		}
		else {
			std::cout << "Saved image to: image.png" << std::endl;
		}

		delete[] image;
	}


}

#endif 