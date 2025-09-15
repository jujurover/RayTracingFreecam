#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "kernel.h"
#include <iostream>
#include <vector>
#include <cmath>
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 1024;

float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
float fov = 45.0f;
int frame = 0;

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

    //update camera
    if(fabs(xoffset) > 1 || fabs(yoffset) > 1) 
    {
        printf("running mouse callback\n");
        xoffset *= sensitivity;
        yoffset *= sensitivity;
        updateCamera(yoffset, xoffset);
        frame = 0;
    }
}

// Scroll wheel callback
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 90.0f)
        fov = 90.0f;
        
    frame = 0;
    // Use fov to update your camera zoom here
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



    //glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, SCR_WIDTH, SCR_HEIGHT);
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 512, 512, GL_RGB, GL_UNSIGNED_BYTE, data);

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

    frame = 0;

    // After window creation:
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        runCuda(frame++, SCR_WIDTH, SCR_HEIGHT);
        display(SCR_WIDTH, SCR_HEIGHT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    cleanupPBO();
    cleanupGraphics();
    return 0;
}



// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        std::cout << "W pressed" << std::endl;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        std::cout << "S pressed" << std::endl;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        std::cout << "A pressed" << std::endl;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        std::cout << "D pressed" << std::endl;
    
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}