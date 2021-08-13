#pragma once

// Standard Library
#include <vector>
#include <iostream>
#include <iomanip>

// OpenGL
#include "glad/glad.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

// Project Headers
#include "Jets_2D/Config.h"

extern bool CameraVectorShow;
extern bool EnableCameraMove;

enum CameraMovement { UP, DOWN, LEFT, RIGHT };

// TODO: Scale pan speed to zoom
const float PAN_SPEED       = 0.1f;         // Movement speed (WASD)
const float ZOOM            = 70.f;         // Perspective distortion
const float FORWARD_SPEED   = 5.f;
const float ZOOM_SPEED      = 1.05f;

const glm::vec3 POSITION_DEFAULT(0.f, 0.f, 2.5f * LIFE_RADIUS);
const glm::vec3 CAMERA_DIRECTION_DEFAULT(0.f, 0.f, -1.f);   // Initially, look straight down Z-Direction into origin
const glm::vec3 UP_DEFAULT(0.f, 1.f, 0.f);                  // Up is Y-Direction, just like typical 2D cartesian

struct camera
{
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;

    float movementSpeed;
    float zoom;

    camera();
    ~camera() { }
    void DefaultView();
    glm::mat4 ViewMatrix();
    void ProcessKeyboard(CameraMovement direction, double deltaTime);
    void ProcessMouseScroll(double yOffset, double deltaTime);
};

extern camera Camera;