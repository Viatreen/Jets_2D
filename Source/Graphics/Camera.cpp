//File Header
#include "Camera.h"

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
#include "Config.h"

camera Camera;

camera::camera() : front(glm::normalize(CAMERA_DIRECTION_DEFAULT)), up(UP_DEFAULT), right(glm::cross(front, up)), position(POSITION_DEFAULT)
{
    movementSpeed = PAN_SPEED;
    zoom = ZOOM;

    this->worldUp = up;
}

void camera::DefaultView()
{
    position = POSITION_DEFAULT;
    front = glm::vec3(CAMERA_DIRECTION_DEFAULT);
}
glm::mat4 camera::ViewMatrix()
{
    return glm::lookAt(position, position + front, up);
}
void camera::ProcessKeyboard(CameraMovement direction, double deltaTime)
{
    // TODO: Add zoom to and from mouse and drag to pan

    float velocity = position.z * PAN_SPEED * float(deltaTime);
    if (direction == UP)
        position += up * velocity;
    if (direction == DOWN)
        position -= up * velocity;
    if (direction == LEFT)
        position -= right * velocity;
    if (direction == RIGHT)
        position += right * velocity;

    //std::cout << std::fixed << std::setprecision(2) << "View Position: " << position.x << ", " << position.y << ", " << position.z << " View Direction: " << front.x << ", " << front.y << ", " << front.z << std::endl;
}

void camera::ProcessMouseScroll(double yOffset, double deltaTime)
{
    // TODO: Don't allow zoom past zero. Slow as closer to 0.

    if (yOffset < 0)
        position.z *= ZOOM_SPEED;
    else
        position.z /= ZOOM_SPEED;
}
