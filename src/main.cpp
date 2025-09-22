#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "kernel.h"
#include <iostream>
#include <vector>
#include <cmath>
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float CAM_SPEED = 25.0f;
const float ROLL_SPEED = 1.0f;
const float LOW_RES_FACTOR = 0.25f; // Factor to reduce resolution when moving

float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
float fov = 40.0f;
int frame = 0;
bool cameraModified = false;

// Mouse movement callback
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed: y ranges bottom to top
    lastX = xpos;
    lastY = ypos;
    
    float sensitivity = 0.05f;

    // Only update camera yaw/pitch if left mouse button is pressed
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        if(fabs(xoffset) > 0.01 || fabs(yoffset) > 0.01) // Add a small threshold to avoid noise
        {
            xoffset *= sensitivity;
            yoffset *= sensitivity;
            stageCameraYawPitch(yoffset, xoffset);
            cameraModified = true;
        }
    }
    else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        firstMouse = true; // Reset for the next time the button is pressed
    }
}

// Scroll wheel callback
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 90.0f)
        fov = 90.0f;
    
    stageCameraFOV(fov);
    cameraModified = true;
    // Use fov to update your camera zoom here
}


void updateViewport()
{
    updateCamera();
    cameraModified = false;
    frame = 0;
}

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    setupScene(SCR_WIDTH, SCR_HEIGHT);
    initPBO(SCR_WIDTH, SCR_HEIGHT); 
    initTexture(SCR_WIDTH, SCR_HEIGHT);
    initScreenQuad();
    initScreenShader();

    int low_res_width = SCR_WIDTH * LOW_RES_FACTOR;
    int low_res_height = SCR_HEIGHT * LOW_RES_FACTOR;

    initPBOLowRes(low_res_width, low_res_height);
    initTextureLowRes(low_res_width, low_res_height);
    initScreenQuadLowRes();

    frame = 0;

    // After window creation:
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);
        glfwPollEvents();

        if(!cameraModified)
        {
            runCuda(frame++, SCR_WIDTH, SCR_HEIGHT);
            display(SCR_WIDTH, SCR_HEIGHT);
        }
        else
        {
            clearHighResBuffer();
            runCudaLowRes(frame++, low_res_width, low_res_height);
            displayLowRes(low_res_width, low_res_height);
            updateViewport();
        }


        glfwSwapBuffers(window);
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    cleanupPBO();
    cleanupPBOLowRes();
    cleanupGraphics();
    cleanupScene();
    return 0;
}



// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        stageCameraMovement(0.0f, 0.0f, -CAM_SPEED);
        cameraModified = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        stageCameraMovement(0.0f, 0.0f, CAM_SPEED);
        cameraModified = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        stageCameraMovement(-CAM_SPEED, 0.0f, 0.0f);
        cameraModified = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        stageCameraMovement(CAM_SPEED, 0.0f, 0.0f);
        cameraModified = true;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        stageCameraMovement(0.0f, CAM_SPEED, 0.0f);
        cameraModified = true;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        stageCameraMovement(0.0f, -CAM_SPEED, 0.0f);
        cameraModified = true;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        stageCameraWorldUpRotation(ROLL_SPEED);
        cameraModified = true;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    {
        stageCameraWorldUpRotation(-ROLL_SPEED);
        cameraModified = true;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        stageResetCameraWorldUp();
        cameraModified = true;
    }
    
    /* if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        int width = SCR_WIDTH, height = SCR_HEIGHT;
        std::vector<unsigned char> pixels(width * height * 3);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        // Flip vertically if needed
        stbi_write_png("screenshot.png", width, height, 3, pixels.data(), width * 3);
    } */

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}