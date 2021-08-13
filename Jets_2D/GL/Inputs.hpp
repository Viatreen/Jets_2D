#pragma once

// Standard Library
#include <stdint.h>

// OpenGL
#include "glad/glad.h"
#include "GLFW/glfw3.h"

extern double RenderTime;

namespace Inputs
{
    void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);
    void ScrollCallback(GLFWwindow* window, double xOffset, double yOffset);
    void MouseCallback(GLFWwindow* window, double xPos, double yPos);
    void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    void DoMovement();
}
