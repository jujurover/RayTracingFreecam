#ifndef RTW_STB_IMAGE_H
#define RTW_STB_IMAGE_H

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
    #pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "external/stb_image.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdlib>
#include <iostream>

class rtw_image {
  public:
    __host__ rtw_image() {}

    __host__ rtw_image(const char* image_filename) {
        // Loads image data from the specified file. If the RTW_IMAGES environment variable is
        // defined, looks only in that directory for the image file. If the image was not found,
        // searches for the specified image file first from the current directory, then in the
        // images/ subdirectory, then the _parent's_ images/ subdirectory, and then _that_
        // parent, on so on, for six levels up. If the image was not loaded successfully,
        // width() and height() will return 0.

        auto filename = std::string(image_filename);
        auto imagedir = getenv("RTW_IMAGES");

        // Hunt for the image file in some likely locations.
        if (imagedir && load(std::string(imagedir) + "/" + image_filename)) return;
        if (load(filename)) return;
        if (load("images/" + filename)) return;
        if (load("../images/" + filename)) return;
        if (load("../../images/" + filename)) return;
        if (load("../../../images/" + filename)) return;
        if (load("../../../../images/" + filename)) return;
        if (load("../../../../../images/" + filename)) return;
        if (load("../../../../../../images/" + filename)) return;

        std::cerr << "ERROR: Could not load image file '" << image_filename << "'.\n";
    }

    __host__ rtw_image(const char* image_filename, int restrict_x, int restrict_y, int restrict_width, int restrict_height)
        : restrict_x(restrict_x), restrict_y(restrict_y), restrict_width(restrict_width), restrict_height(restrict_height)
    {
        restricted = true;

        auto filename = std::string(image_filename);
        auto imagedir = getenv("RTW_IMAGES");

        // Hunt for the image file in some likely locations.
        if (imagedir && load(std::string(imagedir) + "/" + image_filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;
        if (load(filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;
        if (load("images/" + filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;
        if (load("../images/" + filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;
        if (load("../../images/" + filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;
        if (load("../../../images/" + filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;
        if (load("../../../../images/" + filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;
        if (load("../../../../../images/" + filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;
        if (load("../../../../../../images/" + filename, restrict_x, restrict_y, restrict_width, restrict_height)) return;

        std::cerr << "ERROR: Could not load image file '" << image_filename << "'.\n";
    }

    __host__ ~rtw_image() {
        delete[] bdata;
        STBI_FREE(fdata);
    }

    __host__ bool load(const std::string& filename) {
        // Loads the linear (gamma=1) image data from the given file name. Returns true if the
        // load succeeded. The resulting data buffer contains the three [0.0, 1.0]
        // floating-point values for the first pixel (red, then green, then blue). Pixels are
        // contiguous, going left to right for the width of the image, followed by the next row
        // below, for the full height of the image.

        auto n = bytes_per_pixel; // Dummy out parameter: original components per pixel
        fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
        if (fdata == nullptr) return false;

        bytes_per_scanline = image_width * bytes_per_pixel;
        convert_to_bytes();
        return true;
    }

    __host__ bool load(const std::string& filename, int restrict_x, int restrict_y, int restrict_width, int restrict_height) {
        // Loads the linear (gamma=1) image data from the given file name. Returns true if the
        // load succeeded. The resulting data buffer contains the three [0.0, 1.0]
        // floating-point values for the first pixel (red, then green, then blue). Pixels are
        // contiguous, going left to right for the width of the image, followed by the next row
        // below, for the full height of the image.

        auto n = bytes_per_pixel; // Dummy out parameter: original components per pixel
        fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
        if (fdata == nullptr) return false;

        bytes_per_scanline = image_width * bytes_per_pixel;

        if (restrict_x < 0 || restrict_y < 0 || restrict_width <= 0 || restrict_height <= 0 ||
            restrict_x + restrict_width > image_width || restrict_y + restrict_height > image_height)
        {
            std::cerr << "ERROR: Invalid image restriction rectangle.\n";
            delete[] bdata;
            STBI_FREE(fdata);
            fdata = nullptr;
            image_width = 0;
            image_height = 0;
            return false;
        }

        // Create a new floating point data array to hold the restricted region.
        auto total_floats = restrict_width * restrict_height * bytes_per_pixel;
        auto* new_fdata = new float[total_floats];

        // Copy over just the pixels in the restricted region.
        for (auto y = 0; y < restrict_height; y++)
            for (auto x = 0; x < restrict_width; x++)
                for (auto c = 0; c < bytes_per_pixel; c++)
                    new_fdata[(y * restrict_width + x) * bytes_per_pixel + c] =
                        fdata[((y + restrict_y) * image_width + (x + restrict_x)) * bytes_per_pixel + c];

        // Replace the original data with the restricted data.
        delete[] bdata;
        STBI_FREE(fdata);
        fdata = new_fdata;
        image_width = restrict_width;
        image_height = restrict_height;
        bytes_per_scanline = image_width * bytes_per_pixel;
        convert_to_bytes();
        return true;
    }

    __device__ __host__ int width()  const { return (fdata == nullptr) ? 0 : image_width; }
    __device__ __host__ int height() const { return (fdata == nullptr) ? 0 : image_height; }

    __device__ const unsigned char* pixel_data(int x, int y) const {
        // Return the address of the three RGB bytes of the pixel at x,y. If there is no image
        // data, returns magenta.
        static unsigned char magenta[] = { 255, 0, 255 };
        if (bdata == nullptr) return magenta;

        x = clamp(x, 0, image_width);
        y = clamp(y, 0, image_height);

        return bdata + y*bytes_per_scanline + x*bytes_per_pixel;
    }

    __host__ rtw_image* copyToDevice()
    {
        rtw_image* d_image;
        cudaMalloc(&d_image, sizeof(rtw_image));
        cudaMemcpy(d_image, this, sizeof(rtw_image), cudaMemcpyHostToDevice);
        if (bdata != nullptr)
        {
            unsigned char* d_bdata;
            int total_bytes = image_width * image_height * bytes_per_pixel;
            cudaMalloc(&d_bdata, total_bytes);
            cudaMemcpy(d_bdata, bdata, total_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_image->bdata), &d_bdata, sizeof(unsigned char*), cudaMemcpyHostToDevice);
        }
        if (fdata != nullptr)
        {
            float* d_fdata;
            int total_bytes = image_width * image_height * bytes_per_pixel * sizeof(float);
            cudaMalloc(&d_fdata, total_bytes);
            cudaMemcpy(d_fdata, fdata, total_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_image->fdata), &d_fdata, sizeof(float*), cudaMemcpyHostToDevice);
        }
        return d_image;
    }

  private:
    const int      bytes_per_pixel = 3;
    float         *fdata = nullptr;         // Linear floating point pixel data
    unsigned char *bdata = nullptr;         // Linear 8-bit pixel data
    int            image_width = 0;         // Loaded image width
    int            image_height = 0;        // Loaded image height
    int            bytes_per_scanline = 0;
    int restrict_x;
    int restrict_y;
    int restrict_width;
    int restrict_height;
    bool restricted = false;

    __device__ __host__ static int clamp(int x, int low, int high) {
        // Return the value clamped to the range [low, high).
        if (x < low) return low;
        if (x < high) return x;
        return high - 1;
    }

    __device__ __host__ static unsigned char float_to_byte(float value) {
        if (value <= 0.0)
            return 0;
        if (1.0 <= value)
            return 255;
        return static_cast<unsigned char>(256.0 * value);
    }

    __device__ __host__ void convert_to_bytes() {
        // Convert the linear floating point pixel data to bytes, storing the resulting byte
        // data in the `bdata` member.

        int total_bytes = image_width * image_height * bytes_per_pixel;
        bdata = new unsigned char[total_bytes];

        // Iterate through all pixel components, converting from [0.0, 1.0] float values to
        // unsigned [0, 255] byte values.

        auto *bptr = bdata;
        auto *fptr = fdata;
        for (auto i=0; i < total_bytes; i++, fptr++, bptr++)
            *bptr = float_to_byte(*fptr);
    }
};

// Restore MSVC compiler warnings
#ifdef _MSC_VER
    #pragma warning (pop)
#endif

#endif