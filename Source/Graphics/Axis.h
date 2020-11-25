#pragma once

// Standard Library
#include <string>

// OpenGL
#include "glad/glad.h"

// Project Headers
#include "Config.h"
#include "ErrorCheck.h"
#include "GL/Shader.h"

enum
{
	X = 0,
	Y
};

namespace AxisVertices
{
	float X[10] = {
		 LIFE_RADIUS,	-LIFE_RADIUS,	// X
		 0.f,			 0.f,			// Y
		 0.f,			 0.f,			// R
		 0.f,			 0.f,			// G
		 1.f,			 1.f			// B
	};

	float Y[10] = {
		0.f,			 0.f,			// X
		LIFE_RADIUS,	-LIFE_RADIUS,	// Ybc
		0.f,			 0.f,			// R
		0.f,			 0.f,			// G
		1.f,			 1.f			// B
	};
}

struct axis		// TODO: Finish this
{
	unsigned int VA, VB;

	axis(float *vertices)
	{
		// Retrieve GL indices and load to GPU memory
		GLCheck(glGenVertexArrays(1, &VA));
		GLCheck(glGenBuffers(1, &VB));
		GLCheck(glBindVertexArray(VA));
		GLCheck(glBindBuffer(GL_ARRAY_BUFFER, VB));
		GLCheck(glBufferData(GL_ARRAY_BUFFER, 2 * 5 * sizeof(float), vertices, GL_STATIC_DRAW));

		// Vertex Positions
		GLCheck(glEnableVertexAttribArray(0));
		GLCheck(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0));

		GLCheck(glEnableVertexAttribArray(1));
		GLCheck(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(1 * 2 * sizeof(float))));
																						   
		// Vertex Colors																   
		GLCheck(glEnableVertexAttribArray(2));											   
		GLCheck(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(2 * 2 * sizeof(float))));
																						   
		GLCheck(glEnableVertexAttribArray(3));											   
		GLCheck(glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(3 * 2 * sizeof(float))));
																						   
		GLCheck(glEnableVertexAttribArray(4));											   
		GLCheck(glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)(4 * 2 * sizeof(float))));
	}

	~axis()
	{
		GLCheck(glDeleteVertexArrays(1, &VA));
		GLCheck(glDeleteBuffers(1, &VB));
	}

	void Draw()
	{
		GLCheck(glBindVertexArray(VA));
		GLCheck(glDrawArrays(GL_LINES, 0, 2));
		GLCheck(glBindVertexArray(0));
	}
};

axis* Axis[2];
