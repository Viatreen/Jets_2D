// File Header
#include "GLSetup.h"

// Standard Library
#include <iostream>
#include <chrono>
#include "GLSetup.h"

// OpenGL
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

// Project Headers
#include "ErrorCheck.h"
#include "Inputs.h"

GLFWwindow* window;

namespace GUI
{
    extern int ProgressHeight;
    extern int SideBarWidth;
}

namespace GL
{
    std::chrono::steady_clock::time_point Timer;

    // Variables and function for aspect ratio compensation to prevent skewing on window resizeing
    int ScreenWidth, ScreenHeight;

    void framebuffer_size_callback(GLFWwindow* window, int width, int height)
    {
        GLCheck(glViewport(0, 0, width, height));
    }

    void Setup()
    {
        // Init GLFW
        glfwInit();

        glfwSetErrorCallback(glfwErrorCallback);

        // Set GLFW window properties
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);                      // OpenGL version 4.3
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        //glfwWindowHint(GLFW_MAXIMIZED, GL_TRUE); 
        glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_DECORATED, GL_TRUE);
        glfwWindowHint(GLFW_FOCUSED, GL_TRUE);

        // Create OpenGL window context using GLFW. Resolution is arbitrary because window will be maximized soon.
        window = glfwCreateWindow(1920, 1080, "Controls", nullptr, nullptr); // Windowed
        if (nullptr == window)
        {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            system("Pause");
        }

        glfwMakeContextCurrent(window);

        // Set mouse to normal
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        glfwSetKeyCallback(window, Inputs::KeyCallback);

        // Maximize GLFW window
        glfwShowWindow(window);
        glfwMaximizeWindow(window);

        // Initialize GLAD in order tu use GL function calls
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cout << "Failed to initialize GLAD" << std::endl;
            system("Pause");
        }

        // Assign GLFW function callbacks
        //glfwSetKeyCallback(window, Inputs::KeyCallback);
        //glfwSetCursorPosCallback(window, Inputs::MouseCallback);
        //glfwSetScrollCallback(window, Inputs::ScrollCallback);
        //glfwSetMouseButtonCallback(window, Inputs::MouseButtonCallback);
        //glfwSetFramebufferSizeCallback(window, GL::framebuffer_size_callback);

        // Get and set the frame size
        glfwGetFramebufferSize(window, &ScreenWidth, &ScreenHeight);
        GLCheck(glViewport(0, 0, ScreenWidth, ScreenHeight));

        // GL drawing parameters
        GLCheck(glDepthFunc(GL_NEVER));
        GLCheck(glEnable(GL_PRIMITIVE_RESTART));
        GLCheck(glPrimitiveRestartIndex(0xFFFFFFFF));
        //GLCheck(glEnable(GL_BLEND));
        //GLCheck(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        //GLCheck(glEnable(GL_DEPTH_TEST));
    }
}
