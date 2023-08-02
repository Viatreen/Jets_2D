cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

if (UNIX)
    # set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)
    if(ARCH_61)
        message(STATUS "Building for CUDA architecture 6.1. GTX 1080 Ti")
        set(CMAKE_CUDA_ARCHITECTURES 61)
    elseif(ARCH_75)
        message(STATUS "Building for CUDA architecture 7.5. GTX 1660 Ti")
        set(CMAKE_CUDA_ARCHITECTURES 75)
    elseif(ARCH_87)
        message(STATUS "Building for CUDA architecture 8.7. Jetson AGX Orin")
        set(CMAKE_CUDA_ARCHITECTURES 87)
        set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
        set(CMAKE_C_COMPILER /usr/bin/gcc)
    else()
        message(STATUS "Must have CUDA architecture of 61, 75, or 87. Feel free to add any that are 60 or higher (cooperative kernels must be supported)")
    endif()
endif()

if(WIN32)
    message(STATUS "Project is building for Windows")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
endif()

project(Jets_2D LANGUAGES CXX C CUDA)
message("Project name: " ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# set(CMAKE_VERBOSE_MAKEFILE ON)    # For debugging

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

if (UNIX)
    set(_GLFW_X11 1)
    add_definitions(-D_GLFW_X11)
    message(STATUS "Using X11 for window creation")
    find_package(X11 REQUIRED)
    add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)
    message(STATUS "Defining IMGUI_IMPL_OPENGL_LOADER_GLAD to set ImGUI to use glad")
endif()

if (WIN32)
    set(_GLFW_WIN32 1)
    message(STATUS "Using Win32 for window creation")
    add_definitions(-D_GLFW_WIN32)
    list(APPEND glfw_PKG_LIBS "-lgdi32")

    if (GLFW_USE_HYBRID_HPG)
        set(_GLFW_USE_HYBRID_HPG 1)
    endif()

    # Workarounds for VS deprecating parts of the standard library
    if (MSVC)
        message(STATUS "Building for Windows with MSVC")
        set(_CRT_SECURE_NO_WARNINGS 1)
        set(CURL_STATICLIB 1)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    endif()
endif()

include_directories(
    lib/
    lib/imgui/
    lib/imgui/backends/
    lib/GLFW/include/
    lib/ImGuiFileDialog/
    .
)

include_directories(${OpenglIncludeSubdir})

set(sources
    # Program
    Jets_2D/main.cu
    
    Jets_2D/ErrorCheck.cpp
    Jets_2D/Config.cpp
    Jets_2D/GL/Inputs.cu
    Jets_2D/GL/GLSetup.cpp
    Jets_2D/GL/Shader.cpp
    Jets_2D/GPGPU/NeuralNet.cu
    Jets_2D/GPGPU/NeuralNet_Eval.cu
    Jets_2D/GPGPU/Epoch.cu
    Jets_2D/GPGPU/GPSetup.cu
    Jets_2D/GPGPU/Helper.cu
    Jets_2D/GPGPU/MapVertexBuffer.cu
    Jets_2D/GPGPU/Match.cu
    Jets_2D/GPGPU/Physic.cu
    Jets_2D/GPGPU/Round.cu
    Jets_2D/GPGPU/SetVariables.cu
    Jets_2D/GPGPU/Vertices.cu
    Jets_2D/Graphics/Axis.cpp
    Jets_2D/Graphics/Camera.cpp
    Jets_2D/Graphics/Circle.cu
    Jets_2D/Graphics/CircleOfLife.cu
    Jets_2D/Graphics/Component.cu
    Jets_2D/Graphics/GrSetup.cu
    Jets_2D/Graphics/Thrust.cu
    Jets_2D/GUI/GUI.cu
    Jets_2D/GUI/Print_Data_Info.cu
    
    # Glad
    lib/glad/glad.c

    # Dear ImGUI
    lib/imgui/imgui.cpp
    lib/imgui/imgui_draw.cpp
    lib/imgui/backends/imgui_impl_glfw.cpp
    lib/imgui/backends/imgui_impl_opengl3.cpp
    lib/imgui/imgui_widgets.cpp
    lib/imgui/imgui_tables.cpp
    lib/ImGuiFileDialog/ImGuiFileDialog.cpp

    # GLFW
    lib/GLFW/src/egl_context.h
    lib/GLFW/src/egl_context.c
    lib/GLFW/src/osmesa_context.h
    lib/GLFW/src/osmesa_context.c
    lib/GLFW/src/window.c
    lib/GLFW/src/init.c
    lib/GLFW/src/input.c
    lib/GLFW/src/monitor.c
    lib/GLFW/src/context.c
    lib/GLFW/src/vulkan.c
)

if(MSVC)
    set(CMAKE_VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

add_executable(${CMAKE_PROJECT_NAME} ${sources})

if (WIN32)
    message(STATUS "Defining _WIN32 for vs command-line support")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _WIN32)
endif()

if (_GLFW_X11)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    lib/GLFW/src/x11_platform.h
    lib/GLFW/src/x11_init.c
    lib/GLFW/src/x11_monitor.c
    lib/GLFW/src/x11_window.c
    lib/GLFW/src/posix_time.h
    lib/GLFW/src/posix_thread.h
    lib/GLFW/src/xkb_unicode.h
    lib/GLFW/src/xkb_unicode.c
    lib/GLFW/src/posix_time.c
    lib/GLFW/src/posix_thread.c
    lib/GLFW/src/glx_context.h
    lib/GLFW/src/glx_context.c
    lib/GLFW/src/linux_joystick.c
)
endif()

if (_GLFW_WIN32)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    lib/GLFW/src/win32_platform.h
    lib/GLFW/src/win32_joystick.h
    lib/GLFW/src/win32_init.c
    lib/GLFW/src/win32_joystick.c
    lib/GLFW/src/win32_monitor.c
    lib/GLFW/src/win32_time.c
    lib/GLFW/src/win32_thread.c
    lib/GLFW/src/win32_window.c
    lib/GLFW/src/wgl_context.h
    lib/GLFW/src/wgl_context.c
)
endif()

if (UNIX)
    target_link_libraries(${CMAKE_PROJECT_NAME}
        ${CMAKE_THREAD_LIBS_INIT}
        ${X11_LIBRARIES}
        ${CMAKE_DL_LIBS}
)
endif()

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${OPENGL_LIBRARIES} 
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:C>:   >
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr> # -g -G --ptxas-options=–allow-expensive-optimizations=true>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if(ARCH_61)
    set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE GTX_1080TI)
elseif(ARCH_75)
    set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE GTX_1660TI)
elseif(ARCH_87)
    set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 87)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ORIN)
else()
    message(STATUS "Must have CUDA architecture of 61, 75, or 87")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    message(STATUS "Building in Release mode")
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)# -Xptxas=–allow-expensive-optimizations=true>)
    if(MSVC)
        target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /Oxs>)
    else()
        target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    endif()
endif()
