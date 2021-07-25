#pragma once

// Standard Library
#include <iostream>

// OpenGL
#include "glad/glad.h"

///////////////////////////////////////////
//  This file checks errors for OpenGL
//  API functions
///////////////////////////////////////////

// GLFW Error Handling
void glfwErrorCallback(int i, const char* err_str);

#ifdef _DEBUG       // Visual Studio define for debug mode
#define ERROR_CHECK_GL
#endif

#ifndef ERROR_CHECK_GL
#define ERROR_CHECK_GL
#endif

#ifdef ERROR_CHECK_GL
#ifndef ASSERT_GL
#define ASSERT_GL(x) if (!(x))    ;//   __debugbreak;
#endif

#define GLCheck(x)  GLClearError();\
                    x;\
                    ASSERT_GL(GLLogCall(__FILE__, __LINE__, #x));


void GLClearError();
bool GLLogCall(const char* FileName, int const LineNumber, const char* FunctionName);

#else
#define GLCheck(x) (x)
#endif