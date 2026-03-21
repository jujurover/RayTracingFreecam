cmake_minimum_required(VERSION 3.18)
project(RayTracingFreecam LANGUAGES CXX CUDA)

# Use vcpkg toolchain if available
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find GLFW (vcpkg manifest mode)
find_package(glfw3 CONFIG REQUIRED)

# Find GLAD
find_package(glad CONFIG REQUIRED)

# Executable
add_executable(RayTracingFreecam
    src/main.cpp
    src/kernel.cu
    src/vec3_0.h src/ray.h src/utility.h src/interval.h src/AABB.h src/onb.h src/color.h src/hittable.h src/hittableList.h
    src/background.h src/BVHNode.h src/pdf.h src/texture.h src/material.h src/triangle.h src/sphere.h src/quadrilateral.h
    src/camera.h src/rtw_stb_image.h
)

# Properties
set_target_properties(RayTracingFreecam PROPERTIES
    CXX_STANDARD 20
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Include & link
target_link_libraries(RayTracingFreecam PRIVATE
    CUDA::cudart
    glfw      # <- matches the imported target in glfw3Targets.cmake
    glad::glad
)