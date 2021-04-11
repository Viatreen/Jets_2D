// File Header
#include "Shader.h"

// Standard Library
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// GLAD
#include "glad/glad.h"

// Project Headers
#include "ErrorCheck.h"

void Shader::Create(const std::string FilePath)
{
	//this->FilePath = FilePath;

	std::ifstream FileStream;
	FileStream.exceptions(/*std::ifstream::failbit | */std::ifstream::badbit);
	try
	{
		FileStream.open(FilePath);
		if (!FileStream.is_open())
			std::cout << "Shader file could not be opened..." << std::endl;
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "Exception opening " << FilePath << "(" << e.code() << "): " << e.what() << std::endl;
		std::cin.get();
	}

	enum struct ShaderType
	{
		NONE = -1,
		VERTEX = 0,
		FRAGMENT = 1
	};

	ShaderType shaderType = ShaderType::NONE;
	std::string Line;
	std::stringstream ShaderStream[2];

	try
	{
		while (getline(FileStream, Line))
		{
			if (Line.find("#shader") != std::string::npos)
			{
				if (Line.find("vertex") != std::string::npos)
					shaderType = ShaderType::VERTEX;
				else if (Line.find("fragment") != std::string::npos)
					shaderType = ShaderType::FRAGMENT;
				else
					std::cout << "\'vertex\' or \'fragment\' keyword not found on line containing \'#shader\' indicator" << std::endl;
			}
			else
			{
				ShaderStream[(int)shaderType] << Line << '\n';
			}
		}
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "Error reading file: " << FilePath << ".\n Bad bit exception thrown" << std::endl;
		std::cin.get();
	}

	try
	{
		FileStream.close();
	}
	catch (...)
	{
		std::cout << "Application error closing shader file: " << FilePath << std::endl;
	}

	//// Compile shaders
	unsigned int ShaderVertexIndex, ShaderFragmentIndex;
	int SuccessFlag;
	char infoLog[512];

	// Vertex Shader
	try
	{
		ShaderVertexIndex = glCreateShader(GL_VERTEX_SHADER);
		std::string ShaderVertexString = ShaderStream[(int)ShaderType::VERTEX].str();
		const char* pShaderVertexString = ShaderVertexString.c_str();
		GLCheck(glShaderSource(ShaderVertexIndex, 1, &pShaderVertexString, NULL));
		GLCheck(glCompileShader(ShaderVertexIndex));
	}
	catch (const std::exception& e)
	{
		std::cout << "Vertex shader section not found. Exception: " << e.what() << std::endl;
		std::cin.get();
	}

	// Print compile errors if any
	GLCheck(glGetShaderiv(ShaderVertexIndex, GL_COMPILE_STATUS, &SuccessFlag));
	if (!SuccessFlag)
	{
		GLCheck(glGetShaderInfoLog(ShaderVertexIndex, 512, NULL, infoLog));
		std::cout << "Error: Vertex shader compilation failed\n" << infoLog << std::endl;
	}

	// Fragment Shader
	try
	{
		ShaderFragmentIndex = glCreateShader(GL_FRAGMENT_SHADER);
		std::string ShaderFragmentString = ShaderStream[(int)ShaderType::FRAGMENT].str();
		const char* pShaderFragmentString = ShaderFragmentString.c_str();
		GLCheck(glShaderSource(ShaderFragmentIndex, 1, &pShaderFragmentString, NULL));
		GLCheck(glCompileShader(ShaderFragmentIndex));
	}
	catch (const std::exception& e)
	{
		std::cout << "Fragment shader section not found. Exception: " << e.what() << std::endl;
		std::cin.get();
	}

	// Print compile errors
	GLCheck(glGetShaderiv(ShaderFragmentIndex, GL_COMPILE_STATUS, &SuccessFlag));
	if (!SuccessFlag)
	{
		GLCheck(glGetShaderInfoLog(ShaderFragmentIndex, 512, NULL, infoLog));
		std::cout << "Error: Fragment shader compilation failed\n" << infoLog << std::endl;
	}

	// Shader Program
	this->ID = glCreateProgram();
	GLCheck(glAttachShader(this->ID, ShaderVertexIndex));
	GLCheck(glAttachShader(this->ID, ShaderFragmentIndex));
	GLCheck(glLinkProgram(this->ID));

	// Print linking errors
	GLCheck(glGetProgramiv(this->ID, GL_LINK_STATUS, &SuccessFlag));
	if (!SuccessFlag)
	{
		GLCheck(glGetProgramInfoLog(this->ID, 512, NULL, infoLog));
		std::cout << "Error: Shader linking failed\n" << infoLog << std::endl;
	}

	// Delete uncompiled string of the shader program from GPU memory
	GLCheck(glDeleteShader(ShaderVertexIndex));
	GLCheck(glDeleteShader(ShaderFragmentIndex));
}
