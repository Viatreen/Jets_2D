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
#include "Config.h"

extern bool CameraVectorShow;
extern bool EnableCameraMove;

enum CameraMovement { UP, DOWN, LEFT, RIGHT };

// TODO: Scale pan speed to zoom
const float PAN_SPEED		= 0.1f;			// Movement speed (WASD)
const float ZOOM			= 70.f;			// Perspective distortion
const float FORWARD_SPEED	= 5.f;
const float ZOOM_SPEED		= 1.05f;

const glm::vec3 POSITION_DEFAULT(0.f, 0.f, 2.5f * LIFE_RADIUS);
const glm::vec3 CAMERA_DIRECTION_DEFAULT(0.f, 0.f, -1.f);	// Initially, look straight down Z-Direction into origin
const glm::vec3 UP_DEFAULT(0.f, 1.f, 0.f);					// Up is Y-Direction, just like typical 2D cartesian


struct camera
{
	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 worldUp;

	float movementSpeed;
	float zoom;

	camera() : front(glm::normalize(CAMERA_DIRECTION_DEFAULT)), up(UP_DEFAULT), right(glm::cross(front, up)), position(POSITION_DEFAULT)
	{
		movementSpeed = PAN_SPEED;
		zoom = ZOOM;

		this->worldUp = up;
	}

	~camera() {	}
	void DefaultView()
	{
		position = POSITION_DEFAULT;
		front = glm::vec3(CAMERA_DIRECTION_DEFAULT);
	}
	glm::mat4 ViewMatrix()
	{
		return glm::lookAt(position, position + front, up);
	}
	void ProcessKeyboard(CameraMovement direction, double deltaTime)
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

	void ProcessMouseScroll(double yOffset, double deltaTime)
	{
		// TODO: Don't allow zoom past zero. Slow as closer to 0.
		
		if (yOffset < 0)
			position.z *= ZOOM_SPEED;
		else
			position.z /= ZOOM_SPEED;
	}
};

camera Camera;