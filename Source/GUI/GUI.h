#pragma once

// CUDA
#include "device_launch_parameters.h"

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
#include "Config.h"
#include "GL/GLSetup.h"
#include "GPGPU/GPSetup.h"
#include "GPGPU/SetVariables.h"
#include "GPGPU/State.h"
#include "GPGPU/GPErrorCheck.h"

// Used in SaveCSV() and Load()
struct CraftWeights
{
	float w[WEIGHT_COUNT];
};

extern CraftWeights* h_CraftWeights;
extern CraftWeights* d_CraftWeights;

extern  bool ShowProgress;
extern  bool ShowSideBar;
extern  bool ShowStateBar;
extern  bool ShowSideBarMutation;
extern  bool RenderAll;
extern  bool RenderFit;
extern  bool RenderOne;
extern  bool RenderNone;
extern  bool SimulateFastFlag;
extern  bool SimulationSpeedToggle;
extern  bool SaveFlag;
extern  bool SaveFlagEndRound;
extern  int SaveCount;
extern  bool LoadBinaryFlag;
extern  bool LoadBinaryFlagEndMatch;
extern  bool LoadBinaryFlagEndRound;

extern ImVec4 clear_color;

extern float TimerStartup;
extern std::chrono::steady_clock::time_point TimerSinceStart;

extern int StepNumber;
extern int MatchNumber;
extern int RoundNumber;
extern float HighScore;
extern int IndexHighScore;
extern float HighScoreCumulative;
extern float HighScoreCumulativeAllTime;
extern std::vector<float> HighScoreVec;
extern std::vector<float> HighScoreVecReverse;

extern std::vector<float> HighScoreCumulativeVec;
extern std::vector<float> HighScoreCumulativeVecReverse;

extern int OpponentRankRange;

// Define Progress plot parameters
extern const int MenuHeight;
extern const int ProgressHeightDefault;
extern const int SideBarWidthDefault;
extern const int StateBarWidthDefault;

extern int ProgressHeight;
extern int SideBarWidth;
extern int StateBarWidth;

extern float ProgressDataWidth;
extern int ProgressPlotOffset;

extern bool MutationChangePending;
extern bool PerformanceChangePending;

extern ImGuiWindowFlags WindowFlags;


extern int LoadCraftCount;
extern bool LoadSuccess;

extern std::vector<std::string> NeuronInputString;
extern std::vector<std::string> NeuronOutputString;

extern bool Pause;

__global__ void SaveWeights(CraftWeights* Weights, CraftState* Crafts, int IndexFrom);  // Kernel called from Save()
__global__ void LoadWeights(CraftWeights* Weights, CraftState* Crafts, int IndexTo);    // Kernel called from Load()
__global__ void CopyState(CraftState* C, state* State, int Index);

// Save data of all crafts in readable format (.csv)
std::string ApplicationRuntime();
void SaveCSV();
void SaveTopBinary(int CraftCount);
void LoadTopBinary1();		// TODO: Tidy up save and load
void LoadTopBinary2();		// TODO: Fix loading issue
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
