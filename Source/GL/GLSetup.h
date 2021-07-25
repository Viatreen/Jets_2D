#pragma once

// Standard Library
#include <chrono>

// OpenGL
#include "glad/glad.h"
#include "GLFW/glfw3.h"

// Project Headers
#include "Inputs.h"

extern GLFWwindow *window;

namespace GUI
{
    extern int ProgressHeight;
    extern int SideBarWidth;
}

namespace GL
{
    extern std::chrono::steady_clock::time_point Timer;

    // Variables and function for aspect ratio compensation to prevent skewing on window resizeing
    extern int ScreenWidth, ScreenHeight;

    void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    void Setup();
}
