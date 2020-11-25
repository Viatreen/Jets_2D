cmake_minimum_required(VERSION 3.10.2)

project(Controls.app LANGUAGES CXX C CUDA)
message("Project name: " ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(X11 REQUIRED)

include_directories(
    lib/
    lib/dear_imgui
    lib/GLFW/include
    lib/qu3e
    Source/
    )

# # Find OpenGL on Linux, Windows, Mac
# find_path(OpenglIncludeSubdir
#     NAMES gl.h
#     PATHS ${OPENGL_INCLUDE_DIR}
#     PATH_SUFFIXES GL OpenGL
#     NO_DEFAULT_PATH
#     )

include_directories(${OpenglIncludeSubdir})

add_definitions(-D_GLFW_USE_CONFIG_H)
add_definitions(-D_DEBUG)

set(sources
    # Glad
    lib/glad/glad.c

    # Program
    Source/main.cu

    # Dear ImGUI
    lib/imgui/imgui.cpp
    lib/imgui/imgui_draw.cpp
    lib/imgui/imgui_impl_glfw.cpp
    lib/imgui/imgui_impl_opengl3.cpp
    lib/imgui/imgui_widgets.cpp    

    # GLFW
    lib/GLFW/src/context.c
    lib/GLFW/src/egl_context.c
    lib/GLFW/src/glx_context.c
    lib/GLFW/src/init.c
    lib/GLFW/src/input.c
    lib/GLFW/src/linux_joystick.c
    lib/GLFW/src/monitor.c
    lib/GLFW/src/osmesa_context.c
    lib/GLFW/src/posix_thread.c
    lib/GLFW/src/posix_time.c
    lib/GLFW/src/vulkan.c
    lib/GLFW/src/window.c
    lib/GLFW/src/x11_init.c
    lib/GLFW/src/x11_monitor.c
    lib/GLFW/src/x11_window.c
    lib/GLFW/src/xkb_unicode.c

    # qu3e
    lib/qu3e/common/q3Geometry.cpp
    lib/qu3e/common/q3Memory.cpp 
    lib/qu3e/dynamics/q3Body.cpp 
    lib/qu3e/dynamics/q3Contact.cpp
    lib/qu3e/dynamics/q3ContactManager.cpp
    lib/qu3e/dynamics/q3ContactSolver.cpp
    lib/qu3e/dynamics/q3Island.cpp
    lib/qu3e/math/q3Mat3.cpp
    lib/qu3e/math/q3Quaternion.cpp
    lib/qu3e/math/q3Vec3.cpp
    lib/qu3e/scene/q3Scene.cpp
    lib/qu3e/collision/q3Box.cpp
    lib/qu3e/collision/q3Collide.cpp
    lib/qu3e/broadphase/q3BroadPhase.cpp
    lib/qu3e/broadphase/q3DynamicAABBTree.cpp
    )

#target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _GLFW_USE_CONFIG_H)

add_executable(${CMAKE_PROJECT_NAME} ${sources})

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:C>:    -g -std=c99>
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:  -g -std=c++17>
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -G -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
    #PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -G -arch=sm_72 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>
    )

if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "-std=c++17")
    set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
endif()


#set_property(TARGET ${CMAKE_PROJECT_NAME}
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${OPENGL_LIBRARIES}
    # glfw
    )

if (UNIX)
    target_link_libraries(${CMAKE_PROJECT_NAME}
        ${CMAKE_THREAD_LIBS_INIT}
        ${X11_LIBRARIES}
        ${CMAKE_DL_LIBS}
        )
endif()
