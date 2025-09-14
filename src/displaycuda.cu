//#include <glad/glad.h>
//#include <GLFW/glfw3.h>
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
//#include <device_launch_parameters.h>
//
//GLuint pbo = 0;
//struct cudaGraphicsResource* cudaPboResource;
//void initPBO() {
//    // Create pixel buffer object
//    glGenBuffers(1, &pbo);
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
//    glBufferData(GL_PIXEL_UNPACK_BUFFER, 800 * 600 * 4, 0, GL_DYNAMIC_DRAW);
//
//    // Register PBO with CUDA
//    cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo,
//        cudaGraphicsRegisterFlagsWriteDiscard);
//}
//void cleanupPBO()
//{
//    cudaGraphicsUnregisterResource(cudaPboResource);
//    glDeleteBuffers(1, &pbo);
//}
//__global__ void renderKernelTest(uchar4* pixels, int width, int height, int frame)
//{
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if (x >= width || y >= height) return;
//
//    int idx = y * width + x;
//    unsigned char r = (x + frame) % 256;
//    unsigned char g = (y + frame) % 256;
//    unsigned char b = 128;
//    pixels[idx] = make_uchar4(r, g, b, 255);
//}
//
//void runCuda(int frame)
//{
//    uchar4* devPtr;
//    size_t size;
//    cudaGraphicsMapResources(1, &cudaPboResource, 0);
//    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource);
//
//    dim3 block(16, 16);
//    dim3 grid((800 + block.x - 1) / block.x,
//        (600 + block.y - 1) / block.y);
//
//    renderKernelTest << <grid, block >> > (devPtr, 800, 600, frame);
//    cudaDeviceSynchronize();
//
//    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
//}