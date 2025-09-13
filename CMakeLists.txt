cmake_minimum_required(VERSION 3.10.0)
project(RayTracingFreecam VERSION 0.1.0 LANGUAGES C CXX CUDA)

add_executable(RayTracingFreecam main.cpp)

find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

target_link_libraries(RayTracingFreecam PRIVATE glad::glad glfw)