// File Header
#include "Jets_2D/ErrorCheck.hpp"

// Standard Library
#include <iostream>

// OpenGL
#include "glad/glad.h"

// GLFW Error Handling
void glfwErrorCallback(int i, const char* err_str)
{
    std::cout << "GLFW Error(" << i << "): " << err_str << std::endl;
    std::cin.get();
}

void GLClearError()
{
    while (glGetError() != GL_NO_ERROR);
}

bool GLLogCall(const char* FileName, int const LineNumber, const char* FunctionName)
{
    while (GLenum error = glGetError())
    {
        std::cout << "[OpenGL Error]: (" << error << "): " << FileName << ":" << LineNumber << " " << FunctionName << std::endl;
        return false;
    }
    return true;
}