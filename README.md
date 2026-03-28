# RayTracingFreecam

*fully interactive scene renderer made with C++ and CUDA*

## Description

### Intro
This is an interactive scene tracer that uses some cool physics and statistics to simulate light propagation, resulting in realistic renders. It is (kind of) parallelized with CUDA (so you will need a NVIDIA gpu to run this). The only external libraries used are those relating to displaying the output in a window. 

### Controls
1. W: forwards
2. A: left
3. S: backwards
4. D: right
5. Q/E: rotate perspective CCW/CW
6. R: reset perspective rotation to 0 degrees
7. Shift: down
8. Space: up

### Rough principles

The camera simulates light rays leaving the camera, propagating its path through the scene as it interacts with scene objects. If the light ray hits an illumination source, that ray's color is set to the illumination source's color scaled by some attenuation factor due to the ray's path (i.e. light absorption by objects, etc.). The simulation is optimized utilizing a surface area heuristic (SAH) bounding volume hierarchy and importance sampling. 


## Some cool renders

here is a nice render I made:

![cornell box with tilted mirror box, glass sphere, and glowstone block](./src/renders/render%203-28-26.png)

[Video showcasing the movement and dynamic rendering of the application](./src/renders/showcase.mp4)

[Time lapse of incremental pixel sampling](./src/renders/sped%20up%20render.mp4)

---

Early test of basic rendering:

![many spheres on an infinite checker plane](./src/renders/checkerboard%20balls%208-2-25.png)

Various test of refraction physics:

![glass cube in cornell box](./src/renders/glass%20cube%20in%20box%208-17-25.png)

![glass sphere in cornell box](./src/renders/glass%20sphere%208-17-25.png)

![glass things with checker box in cornell box](./src/renders/glass%20thing%208-18-25.png)

---

Some progress pics in making a high res background image:

![small divisions background floor test](./src/renders/wallpaper%20progress%202%208-18-25.png)

![large divisions background floor test (with taller divisions)](./src/renders/wallpaper%20progress%208-18-25.png)

![large divisions background floor test](./src/renders/wallpaper%20progress%203%208-19-25.png)

![strange glass things floating in the air](./src/renders/wallpaper%20progress%204%208-19-25.png)

![extremely cluttered glass things in the air](./src/renders/wallpaper%20progress%205%208-19-25.png)

## How to run

Stuff to make sure before running:
- ensure CUDA compiler (must be pre installed) matches Nvidia GPU.
- ensure vcpkg is properly integrated (make sure toolchain file is correctly configured)

When building for the first time, use this command:
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake

Then, use:
cmake --build build

## acknowledgements

Many thanks to "EzRT" by AKGWSB and "Ray Tracing: The Rest of Your Life" by Peter Shirley, Trevor David Black, Steve Hollasch for teaching me the basic principles behind ray tracing. 

