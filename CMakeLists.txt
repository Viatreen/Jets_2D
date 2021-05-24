cmake_minimum_required(VERSION 3.17)

project(Controls.app LANGUAGES CXX C CUDA)
message("Project name: " ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

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
    # TODO: See if we can set VS working directory to cmake root directory
endif()

include_directories(
    lib/
    lib/dear_imgui
    lib/GLFW/include
    Source/
    )

include_directories(${OpenglIncludeSubdir})

if (_GLFW_WIN32)
    add_definitions(-D_GLFW_WIN32)
    list(APPEND glfw_PKG_LIBS "-lgdi32")

    if (GLFW_USE_HYBRID_HPG)
        set(_GLFW_USE_HYBRID_HPG 1)
    endif()
endif()

if (WIN32)
    # Workarounds for VS deprecating parts of the standard library
    if (MSVC)
        set(_CRT_SECURE_NO_WARNINGS 1)
        set(CURL_STATICLIB 1)
    endif()
endif()

set(sources
    # Program
    Source/main.cu
    
    Source/ErrorCheck.cpp
    Source/Config.cpp
    Source/GL/Inputs.cu
    Source/GL/GLSetup.cpp
    Source/GL/Shader.cpp
    Source/GPGPU/NeuralNet.cu
    Source/GPGPU/Epoch.cu
    Source/GPGPU/GPSetup.cu
    Source/GPGPU/MapVertexBuffer.cu
    Source/GPGPU/Match.cu
    Source/GPGPU/Physic.cu
    Source/GPGPU/Round.cu
    Source/GPGPU/SetVariables.cu
    Source/GPGPU/Vertices.cu
    Source/Graphics/Axis.cpp
    Source/Graphics/Camera.cpp
    Source/Graphics/Circle.cu
    Source/Graphics/CircleOfLife.cu
    Source/Graphics/Component.cu
    Source/Graphics/GrSetup.cu
    Source/Graphics/Thrust.cu
    Source/GUI/GUI.cu
    Source/GUI/Print_Data_Info.cu
    
    # Glad
    lib/glad/glad.c

    # Dear ImGUI
    lib/imgui/imgui.cpp
    lib/imgui/imgui_draw.cpp
    lib/imgui/imgui_impl_glfw.cpp
    lib/imgui/imgui_impl_opengl3.cpp
    lib/imgui/imgui_widgets.cpp

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

add_executable(${CMAKE_PROJECT_NAME} ${sources})

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

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:C>:   >
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
)

if(WIN32)
    add_definitions(-D_GLFW_WIN32)
endif()

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr> # --ptxas-options=–allow-expensive-optimizations=true>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60 61 62 70 72 75) # 61 = GTX 1080 Ti

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${OPENGL_LIBRARIES}
    )
