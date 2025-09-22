#include <device_launch_parameters.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include "vec3_0.h"
#include "ray.h"
#include "utility2.h"
#include "interval.h"
#include "AABB.h"
#include "onb.h"
#include "color.h"
#include "hittable.h"
#include "pdf.h"
#include "texture.h"
#include "material.h"
#include "hittableList.h"
#include "sphere.h"
#include "triangle.h"
#include "quadrilateral.h"
#include "camera.h"
#include "background.h"
#include "BVHNode.h"
#include "rtw_stb_image.h"

#include "kernel.h"


#define CHECK_CUDA_CALL(call)  do {                               \
    cudaError_t err__ = (call);                                   \
    if (err__ != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err__));   \
        throw std::runtime_error("CUDA error");                   \
    }                                                             \
} while(0)

//device side constructors/helpers
__global__ void constructMaterialKernel(Material* m, bool emissive, color mat_color, float probability_illumination, float brightness, float specularRate, float roughness, float refractRate, float indexOfRefraction, float refract_roughness) {
    new (m) Material(emissive, mat_color, probability_illumination, brightness, specularRate, roughness, refractRate, indexOfRefraction, refract_roughness);
}
__global__ void constructMaterialKernel(Material* m, bool emissive, texture* tex, float probability_illumination, float brightness, float specularRate, float roughness, float refractRate, float indexOfRefraction, float refract_roughness) {
    new (m) Material(emissive, tex, probability_illumination, brightness, specularRate, roughness, refractRate, indexOfRefraction, refract_roughness);
}
__global__ void constructSolidColorKernel(solid_color** sc_ptr, color c) {
    *sc_ptr = new solid_color(c);
}
__global__ void constructCheckerTextureKernel(checker_texture** ct_ptr, texture* even, texture* odd, float scale) {
    *ct_ptr = new checker_texture(scale, even, odd);
}
__global__ void constructImageTextureKernel(image_texture** it_ptr, rtw_image* img) {
    *it_ptr = new image_texture(img);
}
__global__ void constructImageKernel(image_texture** it_ptr, rtw_image* img) {
    *it_ptr = new image_texture(img);
}
__global__ void constructSphereKernel(sphere** s_ptr, point3 c, float r, Material* m) {
    *s_ptr = new sphere(c, r, m);
}
__global__ void constructTriangleKernel(triangle** t_ptr, point3 p1, point3 p2, point3 p3, Material* m) {
    *t_ptr = new triangle(p1, p2, p3, m);
}
__global__ void constructQuadrilateralKernel(quadrilateral** q_ptr, point3 Q, vec3 u, vec3 v, Material* m) {
    *q_ptr = new quadrilateral(Q, u, v, m);
}
__global__ void constructGradientBackgroundKernel(gradient_background** bg_ptr, color color1, color color2, color illumination_color)
{
    *bg_ptr = new gradient_background(color1, color2, illumination_color);
}
__global__ void constructSolidBackgroundKernel(solid_background** bg_ptr, color bg_color, color illumination_color)
{
    *bg_ptr = new solid_background(bg_color, illumination_color);
}
__global__ void rotateObjectKernel(rotator** d_rotated, hittable** d_object, float pitch, float roll, float yaw)
{
    *d_rotated = new rotator(*d_object, pitch, roll, yaw);
}
__global__ void constructListKernel(hittable_list** lst_ptr, hittable** obj_array, int num_objects) {
    // Allocate the list itself on the device
    *lst_ptr = new hittable_list();
    (*lst_ptr)->capacity = num_objects;
    (*lst_ptr)->num_objects = 0;

    // Assign the preallocated object array
    (*lst_ptr)->objects = obj_array;

}
__global__ void listAddKernel(hittable_list** lst_ptr, hittable** obj, AABB* bbox) {
    (*lst_ptr)->add(*obj);
    *bbox = (*obj)->bounding_box();
}
__global__ void constructCameraKernel(camera_device* cam, int imageWidth, int imageHeight, int maxDepth, float verticalFov, point3 lookFrom, point3 lookAt, vec3 worldUpVector, float defocus_angle, background** bg)
{
    new (cam) camera_device();
    cam->imageWidth = imageWidth;
    cam->imageHeight = imageHeight;
    cam->maxDepth = maxDepth;
    cam->verticalFov = verticalFov;
    cam->lookFrom = lookFrom;
    cam->lookAt = lookAt;
    cam->worldUpVector = worldUpVector;
    cam->defocus_angle = defocus_angle;
    cam->bg = *bg;
    cam->initialize(); // device-side init (as in your original)
}
__global__ void renderKernel(camera_device* cam, uchar4* gl_image, double* d_accumulation_buffer, hittable* world, hittable* lights, BVHNode* search_tree, int frame)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (i >= cam->imageHeight || j >= cam->imageWidth) return;

    int pixel_index = (cam->imageHeight - i -1) * cam->imageWidth + j;

    ray r = cam->get_ray(j, i);
    color accumulated_color = cam->ray_color(r, cam->maxDepth, *world, *lights, *search_tree);

    if (accumulated_color.x() != accumulated_color.x() ||
        accumulated_color.y() != accumulated_color.y() ||
        accumulated_color.z() != accumulated_color.z())
    {
        accumulated_color = color(0, 0, 0); // NaN check
    }

    // Accumulate color in the accumulation buffer
    int accumulation_index = pixel_index * 3;



    if(frame == 0)
    {
        d_accumulation_buffer[accumulation_index + 0] += accumulated_color.x();
        d_accumulation_buffer[accumulation_index + 1] += accumulated_color.y();
        d_accumulation_buffer[accumulation_index + 2] += accumulated_color.z();
    } else 
    {
        d_accumulation_buffer[accumulation_index + 0] = (d_accumulation_buffer[accumulation_index + 0] * frame + accumulated_color.x()) / (frame + 1);
        d_accumulation_buffer[accumulation_index + 1] = (d_accumulation_buffer[accumulation_index + 1] * frame + accumulated_color.y()) / (frame + 1);
        d_accumulation_buffer[accumulation_index + 2] = (d_accumulation_buffer[accumulation_index + 2] * frame + accumulated_color.z()) / (frame + 1);
    }
 

    // Apply gamma correction and convert to [0, 255]
    color gl_color;
    
    gl_color[0] = fminf(fmaxf(pow(d_accumulation_buffer[accumulation_index + 0], 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f);
    gl_color[1] = fminf(fmaxf(pow(d_accumulation_buffer[accumulation_index + 1], 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f);
    gl_color[2] = fminf(fmaxf(pow(d_accumulation_buffer[accumulation_index + 2], 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f);

    gl_image[pixel_index] = make_uchar4(gl_color.x(), gl_color.y(), gl_color.z(), 255);
}
//wrapper functions for adding objects to the scene
static hittable_list** makeList(int num_objects)
{
    hittable_list** d_list_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_list_ptr, sizeof(hittable_list*)));
    hittable** d_list_objects_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_list_objects_ptr, num_objects * sizeof(hittable*)));

    constructListKernel<<<1, 1>>>(d_list_ptr, d_list_objects_ptr, num_objects);
    CHECK_CUDA_CALL(cudaGetLastError());
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_list_ptr;
}
static void addToList(hittable_list** d_list_ptr, hittable** obj_ptr, std::vector<hittable*>& list_hittables, std::vector<AABB>& bboxes)
{
    //make sure list is preallocated for adding objects!!
    AABB* bbox_tmp;
    CHECK_CUDA_CALL(cudaMallocManaged(&bbox_tmp, sizeof(AABB)));
    listAddKernel << <1, 1 >> > (d_list_ptr, obj_ptr, bbox_tmp);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    bboxes.push_back(*bbox_tmp);
    list_hittables.push_back(*obj_ptr);
    cudaFree(bbox_tmp);
}
static camera_device* makeCamera(int image_width, int image_height, int maxDepth, float vFov, point3 lookFrom, point3 lookAt, vec3 worldUp, float defocus, background** d_bg_ptr)
{
    camera_device* d_cam;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_cam, sizeof(camera_device)));
    constructCameraKernel << <1, 1 >> > (
        d_cam,
        /*imageWidth*/ image_width,
        /*imageHeight*/ image_height,
        /*maxDepth*/ maxDepth,
        /*verticalFov*/ vFov,
        /*lookFrom*/ lookFrom,
        /*lookAt*/   lookAt,
        /*worldUp*/  worldUp,
        /*defocus*/  defocus,
        d_bg_ptr
        );
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_cam;
}
static Material* makeMaterial(bool isEmissive = false, color materialColor = color(0, 0, 0), float probability_illumination = 1.0f, float brightness = 4.0f, float specularRate = 0.0f, float roughness = 0.0f, float refractRate = 0.0f, float indexOfRefraction = 1.0f, float refract_roughness = 0.0f)
{
    Material* d_material;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_material, sizeof(Material)));
    constructMaterialKernel << <1, 1 >> > (d_material, isEmissive, materialColor, probability_illumination, brightness, specularRate, roughness, refractRate, indexOfRefraction, refract_roughness);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_material;
}
static Material* makeMaterial(bool isEmissive = false, texture* tex = nullptr, float probability_illumination = 1.0f, float brightness = 4.0f, float specularRate = 0.0f, float roughness = 0.0f, float refractRate = 0.0f, float indexOfRefraction = 1.0f, float refract_roughness = 0.0f)
{
    Material* d_material;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_material, sizeof(Material)));
    constructMaterialKernel << <1, 1 >> > (d_material, isEmissive, tex, probability_illumination, brightness, specularRate, roughness, refractRate, indexOfRefraction, refract_roughness);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_material;
}
static solid_color** makeSolidColor(color c)
{
    solid_color** d_sc_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_sc_ptr, sizeof(solid_color*)));
    constructSolidColorKernel << <1, 1 >> > (d_sc_ptr, c);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_sc_ptr;
}
static checker_texture** makeCheckerTexture(color even, color odd, float scale)
{
    checker_texture** d_ct_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_ct_ptr, sizeof(checker_texture*)));
    solid_color** even_ptr = makeSolidColor(even);
    solid_color** odd_ptr = makeSolidColor(odd);
    constructCheckerTextureKernel << <1, 1 >> > (d_ct_ptr, *even_ptr, *odd_ptr, scale);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_ct_ptr;
}
static image_texture** makeImageTexture(const char* image_filename)
{
    image_texture** d_it_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_it_ptr, sizeof(image_texture*)));
    rtw_image h_img = rtw_image(image_filename);
    //copy image to device
    rtw_image* d_img = h_img.copyToDevice();
    constructImageTextureKernel << <1, 1 >> > (d_it_ptr, d_img);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_it_ptr;
}
static image_texture** makeImageTexture(const char* image_filename, int restrict_x, int restrict_y, int restrict_width, int restrict_height)
{
    image_texture** d_it_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_it_ptr, sizeof(image_texture*)));
    rtw_image h_img = rtw_image(image_filename, restrict_x, restrict_y, restrict_width, restrict_height);
    //copy image to device
    rtw_image* d_img = h_img.copyToDevice();
    constructImageTextureKernel << <1, 1 >> > (d_it_ptr, d_img);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_it_ptr;
}
static sphere** makeSphere(point3 c, float r, Material* m)
{
    sphere** d_sphere_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_sphere_ptr, sizeof(sphere*)));
    constructSphereKernel << <1, 1 >> > (d_sphere_ptr, c, r, m);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_sphere_ptr;
}
static triangle** makeTriangle(point3 p1, point3 p2, point3 p3, Material* mat)
{
    triangle** d_triangle_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_triangle_ptr, sizeof(triangle*)));
    constructTriangleKernel << <1, 1 >> > (d_triangle_ptr, p1, p2, p3, mat);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_triangle_ptr;
}
static quadrilateral** makeQuadrilateral(point3 Q, vec3 u, vec3 v, Material* mat)
{
    quadrilateral** d_quadrilateral_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_quadrilateral_ptr, sizeof(quadrilateral*)));
    constructQuadrilateralKernel << <1, 1 >> > (d_quadrilateral_ptr, Q, u, v, mat);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_quadrilateral_ptr;
}
static hittable_list** makeBox(point3 a, point3 b, Material* m)
{
    hittable_list** d_box_ptr = makeList(6);
    std::vector<hittable*> dummy_list;
    dummy_list.reserve(6);
    std::vector<AABB> dummy_bboxes;
    dummy_bboxes.reserve(6);

    point3 min = point3(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()), std::fmin(a.z(), b.z()));
    point3 max = point3(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()), std::fmax(a.z(), b.z()));

    vec3 dx = vec3(max.x() - min.x(), 0, 0);
    vec3 dy = vec3(0, max.y() - min.y(), 0);
    vec3 dz = vec3(0, 0, max.z() - min.z());

    addToList(d_box_ptr, (hittable**)makeQuadrilateral(min, dy, dx, m), dummy_list, dummy_bboxes); // front
    addToList(d_box_ptr, (hittable**)makeQuadrilateral(max, -dy, -dz, m), dummy_list, dummy_bboxes); // right
    addToList(d_box_ptr, (hittable**)makeQuadrilateral(max, -dx, -dy, m), dummy_list, dummy_bboxes); // back
    addToList(d_box_ptr, (hittable**)makeQuadrilateral(min, dz, dy, m), dummy_list, dummy_bboxes); // left
    addToList(d_box_ptr, (hittable**)makeQuadrilateral(max, -dz, -dx, m), dummy_list, dummy_bboxes); // top
    addToList(d_box_ptr, (hittable**)makeQuadrilateral(min, dz, dx, m), dummy_list, dummy_bboxes); // bottom

    return d_box_ptr;
}
static gradient_background** makeGradientBackground(color color1, color color2)
{
    gradient_background** d_bg_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_bg_ptr, sizeof(gradient_background*)));
    constructGradientBackgroundKernel << <1, 1 >> > (d_bg_ptr, color1, color2, color(1.0, 1.0, 1.0));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_bg_ptr;
}
static solid_background** makeSolidBackground(color bg_color)
{
    solid_background** d_bg_ptr;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_bg_ptr, sizeof(solid_background*)));
    constructSolidBackgroundKernel << <1, 1 >> > (d_bg_ptr, bg_color, color(1.0, 1.0, 1.0));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_bg_ptr;
}


GLuint pbo = 0;
struct cudaGraphicsResource* cudaPboResource;

GLuint pbo_low_res = 0;
struct cudaGraphicsResource* cudaPboResource_low_res;
GLuint tex_low_res = 0;
GLuint quadVAO_low_res = 0, quadVBO_low_res = 0, quadEBO_low_res = 0;


// Globals (add to your globals)
GLuint tex = 0;
GLuint quadVAO = 0, quadVBO = 0, quadEBO = 0;
GLuint screenShader = 0;
camera_device* d_cam_global = nullptr;
hittable_list** d_world_global = nullptr;
hittable_list** d_lights_global = nullptr;
BVHNode* d_bvh_global = nullptr;
double* d_accumulation_buffer = nullptr;

// Simple shader sources (GLSL 330)
const char* quadVertSrc = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 TexCoord;
void main() {
    TexCoord = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

const char* quadFragSrc = R"glsl(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTex;
void main() {
    FragColor = texture(screenTex, TexCoord);
}
)glsl";

// Helper: compile shader, link program (minimal error printing)
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0'); glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "Shader compile error: " << log << std::endl;
    }
    return s;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0'); glGetProgramInfoLog(p, len, nullptr, log.data());
        std::cerr << "Program link error: " << log << std::endl;
    }
    // detach/delete shaders by caller
    return p;
}

// Call once after GL context creation (before render loop)
void initTexture(int width, int height) {
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    // allocate texture storage RGBA8
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // nearest filtering, clamp to edge
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void initScreenQuad() {
    // fullscreen quad (two triangles)
    float quadVertices[] = {
        // positions   // texcoords
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };
    unsigned int indices[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);

    glBindVertexArray(quadVAO);

    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // pos (location=0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // tex (location=1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void initTextureLowRes(int width, int height) {
    glGenTextures(1, &tex_low_res);
    glBindTexture(GL_TEXTURE_2D, tex_low_res);

    // allocate texture storage RGBA8
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // nearest filtering, clamp to edge
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void initScreenQuadLowRes() {
    // fullscreen quad (two triangles)
    float quadVertices[] = {
        // positions   // texcoords
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };
    unsigned int indices[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &quadVAO_low_res);
    glGenBuffers(1, &quadVBO_low_res);
    glGenBuffers(1, &quadEBO_low_res);

    glBindVertexArray(quadVAO_low_res);

    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_low_res);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO_low_res);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // pos (location=0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // tex (location=1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void initScreenShader() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, quadVertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, quadFragSrc);
    screenShader = linkProgram(vs, fs);
    // we can delete the compiled shader objects after linking
    glDeleteShader(vs);
    glDeleteShader(fs);

    glUseProgram(screenShader);
    // bind the sampler to texture unit 0
    GLint loc = glGetUniformLocation(screenShader, "screenTex");
    if (loc >= 0) glUniform1i(loc, 0);
    glUseProgram(0);
}

// Display: call every frame AFTER runCuda() and AFTER cudaGraphicsUnmapResources
void display(int width, int height)
{
    glClear(GL_COLOR_BUFFER_BIT);

    // Bind PBO as pixel unpack source and upload into texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);

    // Upload from PBO into texture (RGBA)
    // NOTE: your PBO must have size width*height*4 and CUDA should write uchar4 pixels.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // unbind PBO (texture has the data now)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Draw fullscreen quad with the texture
    glUseProgram(screenShader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBindVertexArray(quadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}
void displayLowRes(int width, int height)
{
    glClear(GL_COLOR_BUFFER_BIT);

    // Bind PBO as pixel unpack source and upload into texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_low_res);
    glBindTexture(GL_TEXTURE_2D, tex_low_res);

    // Upload from PBO into texture (RGBA)
    // NOTE: your PBO must have size width*height*4 and CUDA should write uchar4 pixels.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // unbind PBO (texture has the data now)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Draw fullscreen quad with the texture
    glUseProgram(screenShader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_low_res);

    glBindVertexArray(quadVAO_low_res);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

// Call at program exit to free GL resources
void cleanupGraphics() {
    if (tex) { glDeleteTextures(1, &tex); tex = 0; }
    if (quadEBO) { glDeleteBuffers(1, &quadEBO); quadEBO = 0; }
    if (quadVBO) { glDeleteBuffers(1, &quadVBO); quadVBO = 0; }
    if (quadVAO) { glDeleteVertexArrays(1, &quadVAO); quadVAO = 0; }
    if (screenShader) { glDeleteProgram(screenShader); screenShader = 0; }
    if (tex_low_res) { glDeleteTextures(1, &tex_low_res); tex_low_res = 0; }
    if (quadEBO_low_res) { glDeleteBuffers(1, &quadEBO_low_res); quadEBO_low_res = 0; }
    if (quadVBO_low_res) { glDeleteBuffers(1, &quadVBO_low_res); quadVBO_low_res = 0; }
    if (quadVAO_low_res) { glDeleteVertexArrays(1, &quadVAO_low_res); quadVAO_low_res = 0; }
}
void initPBO(int width, int height)
{
    // Create OpenGL Pixel Buffer Object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);

    // Register buffer with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo,
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Error registering PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
void initPBOLowRes(int width, int height)
{
    // Create OpenGL Pixel Buffer Object
    glGenBuffers(1, &pbo_low_res);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_low_res);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);

    // Register buffer with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPboResource_low_res, pbo_low_res,
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Error registering low res PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
void cleanupPBO()
{
    cudaGraphicsUnregisterResource(cudaPboResource);
    glDeleteBuffers(1, &pbo);
}
void cleanupPBOLowRes()
{
    cudaGraphicsUnregisterResource(cudaPboResource_low_res);
    glDeleteBuffers(1, &pbo_low_res);
}
void runCuda(int frame, int width, int height)
{
    uchar4* devPtr;
    size_t size;
    cudaError_t err = cudaGraphicsMapResources(1, &cudaPboResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "Error mapping PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource);
    if (err != cudaSuccess) {
        std::cerr << "Error getting mapped pointer from PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);
    
    d_cam_global->imageWidth = width;
    d_cam_global->imageHeight = height;
    d_cam_global->initialize();

        renderKernel << <grid, block >> > (
        /*camera*/ d_cam_global,
        /*image*/ devPtr,
        /*accumulation buffer*/ d_accumulation_buffer,
        /*world*/ (hittable*)*d_world_global,
        /*lights*/ (hittable*)*d_lights_global,
        /*bvh*/ d_bvh_global,
        /*frame*/ frame
        );
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error synchronizing CUDA device: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
    if (err != cudaSuccess) {
        std::cerr << "Error unmapping PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void runCudaLowRes(int frame, int width, int height)
{
    uchar4* devPtr;
    size_t size;
    cudaError_t err = cudaGraphicsMapResources(1, &cudaPboResource_low_res, 0);
    if (err != cudaSuccess) {
        std::cerr << "Error mapping PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource_low_res);
    if (err != cudaSuccess) {
        std::cerr << "Error getting mapped pointer from PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    d_cam_global->imageWidth = width;
    d_cam_global->imageHeight = height;
    d_cam_global->initialize();

    renderKernel << <grid, block >> > (
        /*camera*/ d_cam_global,
        /*image*/ devPtr,
        /*accumulation buffer*/ d_accumulation_buffer,
        /*world*/ (hittable*)*d_world_global,
        /*lights*/ (hittable*)*d_lights_global,
        /*bvh*/ d_bvh_global,
        /*frame*/ frame
        );
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error synchronizing CUDA device: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaGraphicsUnmapResources(1, &cudaPboResource_low_res, 0);
    if (err != cudaSuccess) {
        std::cerr << "Error unmapping PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}



static double* makeImageBuffer(int image_width, int image_height)
{
    //Image buffer
    double* d_image = nullptr;
    CHECK_CUDA_CALL(cudaMalloc(&d_image, image_width * image_height * 3 * sizeof(double)));
    CHECK_CUDA_CALL(cudaMemset(d_image, 0, image_width * image_height * 3 * sizeof(double)));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_image;
}
static BVHNode* makeBVH(std::vector<hittable*>& list_d_hittables, std::vector<AABB>& list_bboxes)
{
    //make a vector of indices
    std::vector<int> indices(list_d_hittables.size());
    for (int i = 0; i < indices.size(); i++) indices[i] = i;
    BVHNode* h_bvh = new BVHNode(list_d_hittables, indices, list_bboxes, 0, list_d_hittables.size());
    h_bvh->copy_to_device();
    BVHNode* d_bvh;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_bvh, sizeof(BVHNode)));
    cudaMemcpy(d_bvh, h_bvh, sizeof(BVHNode), cudaMemcpyHostToDevice);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_bvh;
}
static rotator** rotateObject(hittable** object, float pitch, float roll, float yaw)
{
    rotator** d_rotated;
    CHECK_CUDA_CALL(cudaMallocManaged(&d_rotated, sizeof(rotator*)));
    rotateObjectKernel << <1, 1 >> > (d_rotated, object, pitch, roll, yaw);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    return d_rotated;
}
/* static void save_image(double* d_image_buffer, int image_width, int image_height)
{
    double* h_image = new double[image_width * image_height * 3];
    CHECK_CUDA_CALL(cudaMemcpy(h_image, d_image_buffer, image_width * image_height * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    ImageMaker::imshow(h_image, image_width, image_height);
    delete[] h_image;
} */

void two_balls_test_setup(int width, int height)
{
    cudaDeviceReset();
    cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 4); //4kb stack originally
    
    std::vector<hittable*> list_d_hittables;
    list_d_hittables.reserve(2);
    std::vector<AABB> list_bboxes;
    list_bboxes.reserve(2);

    std::vector<hittable*> fake_list_d_hittables;
    fake_list_d_hittables.reserve(1);
    std::vector<AABB> fake_list_bboxes;
    fake_list_bboxes.reserve(1);

    Material* d_matte = makeMaterial(false, color(1.0, 1.0, 0.0));
    Material* d_glow = makeMaterial(true, color(10.0, 10.0, 10.0));
    Material* mirror = makeMaterial(false, color(1.0, 1.0, 1.0), 1.0f);
    Material* glass = makeMaterial(false, color(1.0, 1.0, 1.0), 0.0f, 0.0f, 1.0f, 1.5f);

    //Material* checker_mat = makeMaterial(false, (texture*)*makeCheckerTexture(color(0.1f, 0.1f, 0.1f), color(0.9f, 0.9f, 0.9f)));
    Material* image_mat = makeMaterial(false, (texture*)*makeImageTexture("texture_images/minecraft_textures.png"));
    Material* minecraft_texture = makeMaterial(false, (texture*)*makeImageTexture("texture_images/minecraft_textures.png", 15*16, 15*16, 16, 16));

    solid_background** d_bg_ptr = makeSolidBackground(color(0.529f, 0.808f, 0.922f));

    //sphere** d_sphere1_ptr = makeSphere(point3(0.0, 0.0, 0.0), 1.0f, minecraft_texture);
    hittable_list** box1 = makeBox(point3(-0.5, -0.5, -0.5), point3(0.5, 0.5, 0.5), minecraft_texture);

    sphere** d_sphere2_ptr = makeSphere(point3(0.0, 4.0, 0.0), 1.0f, d_glow);

    hittable_list** d_world_ptr = makeList(2);
    hittable_list** d_lights_ptr = makeList(1);

    addToList(d_lights_ptr, (hittable**)d_sphere2_ptr, fake_list_d_hittables, fake_list_bboxes);
    addToList(d_world_ptr, (hittable**)box1, list_d_hittables, list_bboxes);
    //adding to lights
    addToList(d_world_ptr, (hittable**)d_lights_ptr, list_d_hittables, list_bboxes);

    //make bvh for world
    BVHNode* d_bvh = makeBVH(list_d_hittables, list_bboxes);

    camera_device* d_cam = makeCamera(
        /*image_width*/ width,
        /*image_height*/ height,
        /*max depth*/ 5,
        /*vertical fov*/ 40.0,
        /*look from point*/ point3(0, 0, -15),
        /*look at point*/ point3(0, 0, -10),
        /*world up vector*/ vec3(0, 1, 0),
        /*defocus angle*/ 0.0,
        /*background object*/ (background**)d_bg_ptr);

    d_cam_global = d_cam;
    d_world_global = d_world_ptr;
    d_lights_global = d_lights_ptr;
    d_bvh_global = d_bvh;
    d_accumulation_buffer = makeImageBuffer(d_cam->imageWidth, d_cam->imageHeight);
}

void cornell_box_setup(int width, int height)
{
    int num_non_emissive = 0;
    int num_emissive = 0;

    std::vector<hittable*> list_d_hittables;
    std::vector<AABB> list_bboxes;

    std::vector<hittable*> fake_list_d_hittables;
    std::vector<AABB> fake_list_bboxes;

    cudaDeviceReset();
    cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 4); //4kb stack originally

    Material* red = makeMaterial(false, color(.65f, .05f, .05f));
    Material* white = makeMaterial(false, color(.73f, .73f, .73f), 0.0f, 0.0f, 1.0f, 0.4f);
    Material* green = makeMaterial(false, color(.12f, .45f, .15f));
    Material* light = makeMaterial(true, color(1.0f, 1.0f, 1.0f), 1.0f, 6.0f);
    Material* glass = makeMaterial(false, color(1.0, 1.0, 1.0), 0.0f, 0.0f, 1.0f, 1.5f);
    Material* air = makeMaterial(false, color(1.0, 1.0, 1.0), 0.0f, 0.0f, 1.0f, 1.0f);
    Material* minecraft_texture = makeMaterial(true, (texture*)*makeImageTexture("texture_images/minecraft_textures.png", 9*16, 6*16, 16, 16), 0.8f, 6.0f);
    Material* image_mat = makeMaterial(false, (texture*)*makeImageTexture("texture_images/steve_head.jpg"));

    Material* mirror = makeMaterial(false, color(1.0, 1.0, 1.0), 1.0f, 0.0f, 1.0f, 0.0f);

    solid_background** bg = makeSolidBackground(color(color(0.0, 0.0, 0.0)));

    quadrilateral** wall_left = makeQuadrilateral(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), green); num_non_emissive++;
    quadrilateral** wall_right = makeQuadrilateral(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red); num_non_emissive++;
    quadrilateral** floor = makeQuadrilateral(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), white); num_non_emissive++;    
    quadrilateral** wall_back = makeQuadrilateral(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white); num_non_emissive++;
    quadrilateral** ceiling = makeQuadrilateral(point3(0, 0, 555), vec3(0, 555, 0), vec3(555, 0, 0), white); num_non_emissive++;

    quadrilateral** ceiling_light = makeQuadrilateral(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light); num_emissive++;

    //hittable_list** box1 = makeBox(point3(130, 0, 65), point3(295, 165, 230), minecraft_texture); num_emissive++;
    sphere** box1 = makeSphere(point3(190, 90, 190), 90, minecraft_texture); num_emissive++;

    hittable_list** box2 = makeBox(point3(265, 0, 295), point3(430, 330, 460), mirror); num_non_emissive++;
    rotator** box2_rotated = rotateObject((hittable**)box2, 10, 15, 20);

    //hittable_list** box3 = makeBox(point3(300, 35, 330), point3(395, 295, 425), white); num_non_emissive++;

    hittable_list** d_world = makeList(num_non_emissive + num_emissive);
    hittable_list** d_lights = makeList(num_emissive);

    addToList(d_lights, (hittable**)ceiling_light, fake_list_d_hittables, fake_list_bboxes);
    addToList(d_lights, (hittable**)box1, fake_list_d_hittables, fake_list_bboxes); 

    addToList(d_world, (hittable**)wall_left, list_d_hittables, list_bboxes);
    addToList(d_world, (hittable**)wall_right, list_d_hittables, list_bboxes);
    addToList(d_world, (hittable**)floor, list_d_hittables, list_bboxes);
    addToList(d_world, (hittable**)wall_back, list_d_hittables, list_bboxes);
    addToList(d_world, (hittable**)ceiling, list_d_hittables, list_bboxes);
    addToList(d_world, (hittable**)ceiling_light, list_d_hittables, list_bboxes);
    addToList(d_world, (hittable**)box1, list_d_hittables, list_bboxes);
    addToList(d_world, (hittable**)box2_rotated, list_d_hittables, list_bboxes);
    //addToList(d_world, (hittable**)box3, list_d_hittables, list_bboxes);



    //make bvh for world
    BVHNode* d_bvh = makeBVH(list_d_hittables, list_bboxes);

    camera_device* d_cam = makeCamera(
        /*image_width*/ width,
        /*image_height*/ height,
        /*max depth*/ 10,
        /*vertical fov*/ 40.0,
        /*look from point*/ point3(278, 278, -800),
        /*look at point*/ point3(278, 278, 0),
        /*world up vector*/ vec3(0, 1, 0),
        /*defocus angle*/ 0.0,
        /*background object*/ (background**)bg);
    
    d_cam_global = d_cam;
    d_world_global = d_world;
    d_lights_global = d_lights;
    d_bvh_global = d_bvh;
    d_accumulation_buffer = makeImageBuffer(d_cam->imageWidth, d_cam->imageHeight);
}

void cleanupScene()
{
    cudaDeviceReset();
}

void setupScene(int width, int height)
{
    cornell_box_setup(width, height);
    //two_balls_test_setup(width, height);
}



void stageCameraYawPitch(float pitch, float yaw)
{
    d_cam_global->updatePitchYaw(pitch, yaw);
}
void stageCameraFOV(float fov)
{
    d_cam_global->verticalFov = fov;
}
void stageCameraMovement(float right, float up, float forward)
{
    d_cam_global->doCameraMovement(right, up, forward);
}
void stageResetCameraWorldUp()
{
    d_cam_global->worldUpVector = d_cam_global->originalUpVector;
}
void stageCameraWorldUpRotation(float angle)
{
    d_cam_global->moveWorldUpVector(angle);
}
void updateCamera()
{
    d_cam_global->initialize();
    //reset accumulation buffer
    CHECK_CUDA_CALL(cudaMemset(d_accumulation_buffer, 0, d_cam_global->imageWidth * d_cam_global->imageHeight * 3 * sizeof(double)));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
}
void clearHighResBuffer()
{
    CHECK_CUDA_CALL(cudaMemset(d_accumulation_buffer, 0, d_cam_global->imageWidth * d_cam_global->imageHeight * 3 * sizeof(double)));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
}