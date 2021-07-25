#include "Inputs.h"

// Standard Library
#include <vector>
#include <iostream>
#include <stdint.h>

// OpenGL
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

// ImGui
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

// Project Headers
#include "Graphics/Camera.h"
#include "Config.h"
#include "GPGPU/SetVariables.h"

double RenderTime = 0.0;

namespace Inputs
{
    // User input
    static bool keys[1024] = { 0 };

    void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
    {
        if (key >= 0 && key < 1024)
        {
            if (action == GLFW_PRESS)
            {
                keys[key] = true;
            }
            else if (action == GLFW_RELEASE)
            {
                keys[key] = false;
            }
        }

        if (keys[GLFW_KEY_ESCAPE])
        {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
        if (keys[GLFW_KEY_LEFT_CONTROL] || keys[GLFW_KEY_RIGHT_CONTROL])
        {
            /*if (keys[GLFW_KEY_S])
                IO::SaveFlag = true;
            if (keys[GLFW_KEY_L])
                IO::LoadBinaryFlag = true;*/
        }

        if (keys[GLFW_KEY_KP_ADD])
        {
            if ((keys[GLFW_KEY_M] & keys[GLFW_KEY_C])) { h_Config->MutationScaleChance += 0.001f;   SyncConfigArray(); }
            if ((keys[GLFW_KEY_M] & keys[GLFW_KEY_A])) { h_Config->MutationScale += 0.001f;         SyncConfigArray(); }
            if ((keys[GLFW_KEY_M] & keys[GLFW_KEY_S])) { h_Config->MutationFlipChance += 0.0001f;   SyncConfigArray(); }
            if ((keys[GLFW_KEY_W] & keys[GLFW_KEY_M])) { h_Config->WeightMax += 0.1f;               SyncConfigArray(); }
        }
        if (keys[GLFW_KEY_KP_SUBTRACT] == true)
        {
            if (keys[GLFW_KEY_M] & keys[GLFW_KEY_C]) { h_Config->MutationScaleChance -= 0.001f; SyncConfigArray(); }
            if (keys[GLFW_KEY_M] & keys[GLFW_KEY_A]) { h_Config->MutationScale = 0.001f;        SyncConfigArray(); }
            if (keys[GLFW_KEY_M] & keys[GLFW_KEY_S]) { h_Config->MutationFlipChance = 0.0001f;  SyncConfigArray(); }
            if (keys[GLFW_KEY_W] & keys[GLFW_KEY_M]) { h_Config->WeightMax -= 0.1f;             SyncConfigArray(); }
        }
    }

    void ScrollCallback(GLFWwindow* window, double xOffset, double yOffset)
    {
        if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
        {
            Camera.ProcessMouseScroll(yOffset, 1.0 / FRAMES_PER_SECOND);    // TODO: Use actual delta time
        }
    }

    void MouseCallback(GLFWwindow* window, double xPos, double yPos)
    {


        // No commands to ImGui
    }

    void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
    {
        if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
        {
            if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
                Camera.DefaultView();
        }
    }

    void DoMovement()
    {
        // TODO: Fix left and right movement
        if (keys[GLFW_KEY_W] || keys[GLFW_KEY_UP])
            Camera.ProcessKeyboard(UP, 1.0 / FRAMES_PER_SECOND);
        if (keys[GLFW_KEY_S] || keys[GLFW_KEY_DOWN])
            Camera.ProcessKeyboard(DOWN, 1.0 / FRAMES_PER_SECOND);
        if (keys[GLFW_KEY_A] || keys[GLFW_KEY_LEFT])
            Camera.ProcessKeyboard(LEFT, 1.0 / FRAMES_PER_SECOND);
        if (keys[GLFW_KEY_D] || keys[GLFW_KEY_RIGHT])
            Camera.ProcessKeyboard(RIGHT, 1.0 / FRAMES_PER_SECOND);
    }
}
