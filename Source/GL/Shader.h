#pragma once

// Standard Library
#include <string>

// GLAD
#include "glad/glad.h"

// Project Headers
#include "ErrorCheck.h"

struct Shader
{
	short int ID;
	//std::string FilePath;

	Shader() {}

	void Create(const std::string FilePath);

	Shader(const std::string& FilePath) { Create(FilePath); }
	~Shader() { GLCheck(glDeleteProgram(ID)); }

	// Use the current shader
	void Bind() const { GLCheck(glUseProgram(this->ID)); }
	void Unbind() const { GLCheck(glUseProgram(0)); }

	// Set uniform
	void SetUniform(const std::string& name) {	}
};
