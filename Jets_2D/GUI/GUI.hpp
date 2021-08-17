#pragma once

// CUDA
#include <device_launch_parameters.h>

// Standard Library
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cerrno>

// Windows
#ifdef _WIN32
#include <Windows.h>
#include <commdlg.h>
#include <cderr.h>
#endif

// ImGui
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

// Boost
//#include "boost/filesystem.hpp"

// Project Headers
#include "Jets_2D/Config.hpp"
#include "Jets_2D/GL/GLSetup.hpp"
#include "Jets_2D/GPGPU/GPSetup.hpp"
#include "Jets_2D/GPGPU/SetVariables.hpp"
#include "Jets_2D/GPGPU/State.hpp"
#include "Jets_2D/GPGPU/GPErrorCheck.hpp"

namespace GUI
{

// Used in SaveCSV() and Load()
struct CraftWeights
{
	float w[WEIGHT_AMOUNT];
};

extern ImVec4 clear_color;
extern float TimerStartup;
extern int RoundNumber;
extern int OpponentRankRange;

// Define Progress plot parameters
extern const int MenuHeight;

extern int ProgressHeight;
extern int SideBarWidth;

extern bool Pause;

__global__ void SaveWeights(CraftWeights* Weights, CraftState* Crafts, int IndexFrom); // Kernel called from Save()
__global__ void LoadWeights(CraftWeights* Weights, CraftState* Crafts, int IndexTo);   // Kernel called from Load()
__global__ void CopyState(CraftState* C, state* State, int Index);

// Save data of all crafts in readable format (.csv)
std::string ApplicationRuntime();
void SaveCSV();
void SaveTopBinary(int CraftCount);
void LoadTopBinary1(); // TODO: Tidy up save and load
void LoadTopBinary2(); // TODO: Fix loading issue
void NeuronStringSpacePrefixer(std::vector<std::string>& Vec, std::string str, int Length);

// Align text of strings that describe neurons
void NeuronStringAdder(std::vector<std::string>& Vec, std::string Suffix, int Value, int LengthNumber, int LengthString);
void Setup();
void MatchEnd();
void RoundEnd();
void RoundEnd2();

//bool show_demo_window = true;

void AddSpaces(std::string& Output, float Input);
void StateBar(bool LeftSide, state* d_State, float AngleStart);
void Run(int OpponentID, int PositionNumber, float AngleStart);
void Shutdown();

} //End namespace GUI
