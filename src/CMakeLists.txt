cmake_minimum_required(VERSION 3.18)
project(RayTracingFreecam LANGUAGES CXX CUDA)

add_executable(RayTracingFreecam RayTracingFreecam.cpp kernel.cu vec3.h ray.h utility.h interval.h AABB.h onb.h color.h hittable.h hittableList.h ImageMaker.h background.h BVHNode.h pdf.h texture.h material.h triangle.h sphere.h quadrilateral.h camera.h)

set_target_properties(RayTracingFreecam PROPERTIES
    CXX_STANDARD 20
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

#set(CMAKE_PREFIX_PATH "./vcpkg_installed/x64-windows")

find_package(CUDAToolkit REQUIRED)
target_include_directories(RayTracingFreecam PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(RayTracingFreecam PRIVATE CUDA::cudart)

find_package(glfw3 CONFIG REQUIRED)
target_include_directories(RayTracingFreecam PRIVATE ${GLFW_INCLUDE_DIRS})
target_link_libraries(RayTracingFreecam PRIVATE glfw)

find_package(glad CONFIG REQUIRED)
target_include_directories(RayTracingFreecam PRIVATE ${GLAD_INCLUDE_DIRS})
target_link_libraries(RayTracingFreecam PRIVATE glad::glad)

#find_package(OpenGL REQUIRED)
#target_include_directories(RayTracingFreecam PRIVATE ${OPENGL_INCLUDE_DIRS})
#target_link_libraries(RayTracingFreecam PRIVATE ${OPENGL_LIBRARIES})