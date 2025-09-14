cmake_minimum_required(VERSION 3.10.0)
project(RayTracingFreecam VERSION 0.1.0 LANGUAGES C CXX CUDA)

add_executable(RayTracingFreecam 
               src/main.cpp 
               src/AABB.h 
               src/background.h
               src/BVHNode.h 
               src/camera.h 
               src/color.h 
               src/display.h 
               src/hittable.h 
               src/hittableList.h 
               src/ImageMaker.h 
               src/interval.h 
               src/kernel.cu 
               src/material.h 
               src/onb.h 
               src/pdf.h 
               src/quadrilateral.h 
               src/ray.h 
               src/sphere.h 
               src/texture.h 
               src/triangle.h
               src/vec3_0.h
               src/utility2.h)

set_target_properties(RayTracingFreecam PROPERTIES
    CXX_STANDARD 20
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

find_package(CUDAToolkit REQUIRED)
target_include_directories(RayTracingFreecam PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(RayTracingFreecam PRIVATE CUDA::cudart)

find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

target_link_libraries(RayTracingFreecam PRIVATE glad::glad glfw)