// File Header
#include "Jets_2D/GUI/GUI.hpp"

// CUDA
#include <device_launch_parameters.h>

// Standard Library
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <string.h>
#include <sstream>
#include <fstream>
#include <iomanip>
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
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

// ImGuiFileDialog
#include "ImGuiFileDialog.h"

// Project Headers
#include "Jets_2D/Config.hpp"
#include "Jets_2D/GL/GLSetup.hpp"
#include "Jets_2D/GPGPU/GPSetup.hpp"
#include "Jets_2D/GPGPU/SetVariables.hpp"
#include "Jets_2D/GPGPU/State.hpp"
#include "Jets_2D/GPGPU/Round.hpp"
#include "Jets_2D/GPGPU/Match.hpp"

extern bool exit_round;

namespace GUI
{

CraftWeights* h_CraftWeights;
CraftWeights* d_CraftWeights;

bool ShowProgress = true;
bool ShowSideBar = true;
bool ShowStateBar = false;
bool ShowSideBarMutation = false;
bool RenderAll = Config_::RenderAllDefault;
bool RenderFit = Config_::RenderFitDefault;
bool RenderOne = Config_::RenderOneDefault;
bool RenderNone = Config_::RenderNoneDefault;
bool SimulateFastFlag = false;
bool SimulationSpeedToggle = Config_::RenderNoneDefault;
bool SaveFlag = false;
bool SaveFlagEndRound = false;
int SaveCount = SAVE_COUNT_DEFAULT;

ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

float TimerStartup = 0.f;
std::chrono::steady_clock::time_point TimerSinceStart = std::chrono::steady_clock::now();

int StepNumber = 0;
int MatchNumber = 0;
int RoundNumber = 1;
float HighScore = 0.f;
int IndexHighScore = 0;
float HighScoreCumulative = 0.f;
float HighScoreCumulativeAllTime = 0.f;
std::vector<float> HighScoreVec;
std::vector<float> HighScoreVecReverse;

std::vector<float> HighScoreCumulativeVec;
std::vector<float> HighScoreCumulativeVecReverse;

int OpponentRankRange = OPPONENT_RANK_RANGE_DEFAULT;

// Define Progress plot parameters
const int MenuHeight = 19;
const int ProgressHeightDefault = 300;
const int SideBarWidthDefault = 500;
const int StateBarWidthDefault = 600;

int ProgressHeight = ProgressHeightDefault;
int SideBarWidth = SideBarWidthDefault;
int StateBarWidth = StateBarWidthDefault;

float ProgressDataWidth = GL::ScreenWidth - 15.f;
int ProgressPlotOffset = 0;

bool MutationChangePending = false;
bool PerformanceChangePending = false;

ImGuiWindowFlags WindowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

std::vector<std::string> NeuronInputString;
std::vector<std::string> NeuronOutputString;

bool Pause = false;

std::string ApplicationRuntime()
{
    // Create string of total application runtime 
    std::stringstream ApplicationRuntimeStream;
    int SecondsSinceStart = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - TimerSinceStart).count();
    ApplicationRuntimeStream
        << std::setw(2) << std::setfill('0') << SecondsSinceStart / 60 / 60 / 24 << "."
        << std::setw(2) << std::setfill('0') << (SecondsSinceStart / 60 / 60) % 24 << "."
        << std::setw(2) << std::setfill('0') << (SecondsSinceStart / 60) % 60 << "."
        << std::setw(2) << std::setfill('0') << SecondsSinceStart % 60;

    return ApplicationRuntimeStream.str();
}

// Kernel called from Save()
__global__ void SaveWeights(CraftWeights* Weights, CraftState* Crafts, int IndexFrom)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < WEIGHT_AMOUNT)
        Weights->w[idx] = Crafts->Weight[CRAFT_COUNT * idx + IndexFrom];
}

// Kernel called from Load()
__global__ void LoadWeights(CraftWeights* Weights, CraftState* Crafts, int IndexTo)
{
    int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (idx < WEIGHT_AMOUNT)
        Crafts->Weight[CRAFT_COUNT * idx + IndexTo] = Weights->w[idx];
}

__global__ void CopyState(CraftState* C, state* State, int Index)   // Must only call 1 thread from kernel call
{
    State->Score = C->Score[Index];

    State->ScoreBullet = C->ScoreBullet[Index];
    State->ScoreTime = C->ScoreTime[Index];
    State->ScoreDistance = C->ScoreDistance[Index] / 1000;
    State->ScoreFuelEfficiency = C->ScoreFuelEfficiency[Index];

    State->ScoreCumulative = C->ScoreCumulative[Index];

    State->PositionX = C->Position.X[Index];
    State->PositionY = C->Position.Y[Index];

    State->VelocityX = C->Velocity.X[Index];
    State->VelocityY = C->Velocity.Y[Index];

    State->AccelerationX = C->Acceleration.X[Index];
    State->AccelerationY = C->Acceleration.Y[Index];

    State->Angle = C->Angle[Index] * 180.f / PI;
    State->AngularVelocity = C->AngularVelocity[Index] * 180.f / PI;
    State->AngularAcceleration = C->AngularAcceleration[Index] * 180.f / PI;

    State->CannonAngle = C->Cannon.Angle[Index] * 180.f / PI;
    State->Active = C->Active[Index];

    State->CannonCommandAngle = C->CannonCommandAngle[Index] * 180.f / PI;
    State->CannonStrength = C->CannonStrength[Index];

    for (int i = 0; i < NEURON_AMOUNT; i++)
        State->Neuron[i] = C->Neuron[i * CRAFT_COUNT * 2 + Index];

    for (int i = 0; i < 4; i++)
    {
        State->EngineAngle[i] = C->Engine[i].Angle[Index] * 180.f / PI;
        State->EngineAngularVelocity[i] = C->Engine[i].AngularVelocity[Index] * 180.f / PI;
        State->EngineAngularAcceleration[i] = C->Engine[i].AngularAcceleration[Index] * 180.f / PI;
        State->EngineThrustNormalized[i] = C->Engine[i].ThrustNormalized[Index];
    }
}

// Save data of all crafts in readable format (.csv)
void SaveCSV()
{
    // Create and timestamp file
    time_t RawTime;
    tm* TimeInfo;
    time(&RawTime);
    TimeInfo = localtime(&RawTime);
    std::cout << "Saving Information of All Crafts" << std::endl;
    std::cout << std::asctime(TimeInfo);

    //boost::filesystem::path Destination = "Saves";
    //boost::filesystem::create_directory(Destination);

    std::stringstream FileNameStream;

    #ifdef _WIN32
        FileNameStream << "Saves\\";
    #endif

    #ifdef __linux
        FileNameStream << "Saves/";
    #endif

    FileNameStream << "Jets_2D_" << TimeInfo->tm_year + 1900 << std::setw(2) << std::setfill('0')
    << TimeInfo->tm_mon + 1 << std::setw(2) << TimeInfo->tm_mday << "_" << std::setw(2) << TimeInfo->tm_hour 
    << std::setw(2) << TimeInfo->tm_min << std::setw(2) << TimeInfo->tm_sec << "_Score_" << std::setprecision(3)
    << std::fixed << HighScoreCumulative << ".csv";

    std::ofstream File;
    File.open(FileNameStream.str());

    if (!File)
    {
        char ErrorMessage[512];
        perror(ErrorMessage);

        std::cout << "Error: Unable to save crafts file. Message: " << ErrorMessage << std::endl;
    }
    else
    {
        std::stringstream SaveStream;

        // Neural Network
        SaveStream << "Number of layers," << LAYER_AMOUNT << "\n";
        SaveStream << "Total Number of Neurons," << NEURON_AMOUNT << "\n";
        SaveStream << "Number of Weights," << WEIGHT_AMOUNT << "\n";
        SaveStream << "Layer Size Array,";
        for (int i = 0; i < LAYER_AMOUNT - 1; i++)
            SaveStream << Config_::LayerSizeArray[i] << ",";
        SaveStream << Config_::LayerSizeArray[LAYER_AMOUNT - 1] << "\n";        // TODO: Add date, time and array of Parent Rank

        // World Parameters
        SaveStream << "Best Score," << HighScoreCumulativeAllTime << "\n";
        SaveStream << "Hours Running," << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - TimerSinceStart).count() / 3600.f << "\n";        // TODO: Add previous duration to current on load
        SaveStream << "Iterations," << RoundNumber << "\n";
        SaveStream << "Number of Crafts," << CRAFT_COUNT << "\n";
        SaveStream << "Mutation Flip Chance," << h_Config->MutationFlipChance << "\n";
        SaveStream << "Mutation Scale Chance," << h_Config->MutationScaleChance << "\n";
        SaveStream << "Mutation Scale Amount," << h_Config->MutationScale << "\n";
        SaveStream << "Mutation Slide Chance," << h_Config->MutationSlideChance << "\n";
        SaveStream << "Mutation Slide Amount," << h_Config->MutationSigma << "\n";

        // Creature distinction
        SaveStream << "Craft Breakdown,";
        for (int i = 0; i < 3; i++)
            SaveStream << ",";
        SaveStream << "Weights, Layer Number-Originating Neuron-Target Neuron\n";

        File << SaveStream.str();

        std::stringstream ColumnDescriptionString;
        ColumnDescriptionString << "Score,";
        ColumnDescriptionString << "Place,";

        for (unsigned short int L = 0; L < LAYER_AMOUNT - 1; L++)       // Layer Index
            for (int T = 0; T < Config_::LayerSizeArray[L + 1]; T++)        // Target Neuron Index
                for (int O = 0; O < Config_::LayerSizeArray[L]; O++)        // Originating Neuron Index
                    ColumnDescriptionString << "W: " << L << "-" << O << "-" << T << ",";

        ColumnDescriptionString << "\n";
        File << ColumnDescriptionString.str();

        std::stringstream CraftString;

        int h_ScoreCumulative[CRAFT_COUNT];
        int h_CraftPlace[CRAFT_COUNT];
        cudaCheck(cudaMemcpy(&h_ScoreCumulative, Crafts->ScoreCumulative, CRAFT_COUNT * sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(h_CraftPlace, Crafts->Place, CRAFT_COUNT * sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheck(cudaDeviceSynchronize());

        for (int i = 0; i < CRAFT_COUNT; i++)
        {
            CraftWeights* d_CraftWeights;
            cudaCheck(cudaMalloc(&d_CraftWeights, sizeof(CraftWeights)));
            cudaCheck(cudaDeviceSynchronize());

            SaveWeights<<<WEIGHT_AMOUNT / BLOCK_SIZE, BLOCK_SIZE>>>(d_CraftWeights, Crafts, i);
            cudaCheck(cudaDeviceSynchronize());

            CraftWeights* h_CraftWeights = new CraftWeights;
            cudaCheck(cudaMemcpy(h_CraftWeights, d_CraftWeights, sizeof(CraftWeights), cudaMemcpyDeviceToHost));

            CraftString << h_ScoreCumulative[i] << ",";
            CraftString << h_CraftPlace[i] << ",";

            for (int j = 0; j < WEIGHT_AMOUNT - 1; j++)
            {
                CraftString << h_CraftWeights->w[j] << ",";
            }

            CraftString << h_CraftWeights->w[WEIGHT_AMOUNT - 1] << "\n";

            delete h_CraftWeights;
            cudaCheck(cudaFree(d_CraftWeights));
            cudaCheck(cudaDeviceSynchronize());
        }

        File << CraftString.str();
        File.close();

        std::cout << "All Craft Information Saved" << std::endl;
    }
}

void SaveTopBinary(int CraftCount)
{
    // Create and timestamp file
    time_t RawTime;
    tm* TimeInfo;
    time(&RawTime);
    TimeInfo = localtime(&RawTime);
    std::cout << "Saving " << SaveCount << " Best Craft's Binary Information. " << WEIGHT_AMOUNT << " Weights" << std::endl;

    //boost::filesystem::path Destination = "Saves";
    //boost::filesystem::create_directory(Destination);

    std::cout << std::asctime(TimeInfo);
    std::stringstream FileNameStream;
    #ifdef _WIN32
        FileNameStream << "Saves\\";
    #endif

    #ifdef __linux
        FileNameStream << "Saves/";
    #endif

    FileNameStream << "Jets_2D_" << TimeInfo->tm_year + 1900 << std::setw(2) << std::setfill('0')
            << TimeInfo->tm_mon + 1 << std::setw(2) << TimeInfo->tm_mday << "_" << std::setw(2) << TimeInfo->tm_hour 
            << std::setw(2) << TimeInfo->tm_min << std::setw(2) << TimeInfo->tm_sec << "_Score_" << std::setprecision(3)
            << std::fixed << HighScoreCumulative << ".craft";

    std::cout << "Filename: " << FileNameStream.str() << std::endl;

    // Write
    std::ofstream File;
    File.open(FileNameStream.str(), std::ios::binary | std::ios::out);

    if (!File)
    {
        perror("The following error occurred");

        std::cout << "Unable to save crafts file" << std::endl;
    }
    else
    {
        int* pCraftCount = &CraftCount;

        int WeightCount = WEIGHT_AMOUNT;
        int* pWeightCount = &WeightCount;

        File.write((char*)pCraftCount, sizeof(int));
        File.write((char*)pWeightCount, sizeof(int));

        for (int i = 0; i < CraftCount; i++)
        {
            CraftWeights* d_CraftWeights;
            cudaCheck(cudaMalloc(&d_CraftWeights, sizeof(CraftWeights)));
            cudaCheck(cudaDeviceSynchronize());

            SaveWeights<<<WEIGHT_AMOUNT / BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_CraftWeights, Crafts, i);
            cudaCheck(cudaDeviceSynchronize());

            CraftWeights* h_CraftWeights = new CraftWeights;
            cudaCheck(cudaMemcpy(h_CraftWeights, d_CraftWeights, sizeof(CraftWeights), cudaMemcpyDeviceToHost));
            cudaCheck(cudaDeviceSynchronize());

            delete h_CraftWeights;
            cudaCheck(cudaFree(d_CraftWeights));
            cudaCheck(cudaDeviceSynchronize());

            File.write((char*)h_CraftWeights, WEIGHT_AMOUNT * sizeof(float));
        }

        std::cout << "All Craft Information Saved" << std::endl;
        File.close();
    }
}

// TODO: Tidy up save and load
void LoadTopBinary(std::string filename)
{
    std::cout << "Opening \"" << filename << "\"" << std::endl;

    int LoadCraftCount;
    bool LoadSuccess = false;

    std::ifstream File;
    File.open(filename.c_str(), std::ios::binary | std::ios::in);
    if (!File)
    {
        std::cout << "Error opening file" << std::endl;
    }
    else
    {
        std::cout << "File opened successfully. Copying queued until after round" << std::endl;

        File.read((char*)&LoadCraftCount, sizeof(int));

        int WeightCount;
        File.read((char*)&WeightCount, sizeof(int));

        if (LoadCraftCount > FIT_COUNT)
            LoadCraftCount = FIT_COUNT;

        if (WeightCount != WEIGHT_AMOUNT)
        {
            std::cout << "Error loading file: Incompatible number of weights in file" << std::endl;
            std::cout << "Target: " << WEIGHT_AMOUNT << ", Source: " << WeightCount << std::endl;
        }
        else
        {
            LoadSuccess = true;

            cudaCheck(cudaMalloc(&d_CraftWeights, LoadCraftCount * sizeof(CraftWeights)));
            cudaCheck(cudaDeviceSynchronize());
            h_CraftWeights = new CraftWeights[LoadCraftCount];

            for (int i = 0; i < LoadCraftCount; i++)
            {
                File.read((char*)h_CraftWeights + i * sizeof(CraftWeights), sizeof(CraftWeights));
                cudaCheck(cudaMemcpy(&d_CraftWeights[i], &h_CraftWeights[i], sizeof(CraftWeights), cudaMemcpyHostToDevice));
            }

            delete h_CraftWeights;
        }

        File.close();
    }
    
    if (LoadSuccess)
    {
        std::cout << "Copying loaded weights" << std::endl;

        for (int i = 0; i < LoadCraftCount; i++)
        {
            LoadWeights<<<WEIGHT_AMOUNT / BLOCK_SIZE + 1, BLOCK_SIZE>>>(&d_CraftWeights[i], Crafts, FIT_COUNT - 1 - i);
            cudaCheck(cudaDeviceSynchronize());
        }

        cudaCheck(cudaFree(d_CraftWeights));
        cudaCheck(cudaDeviceSynchronize());

        std::cout << "Load successful" << std::endl;

        LoadSuccess = false;
    }
}

void NeuronStringSpacePrefixer(std::vector<std::string>& Vec, std::string str, int Length)
{
    while (str.length() < Length)
        str = " " + str;

    Vec.push_back(str);
}

// Align text of strings that describe neurons
void NeuronStringAdder(std::vector<std::string>& Vec, std::string Suffix, int Value, int LengthNumber, int LengthString)
{
    std::string OutString = std::to_string(Value);
    // This function is only run at setup, so don't worry about efficiency
    // Prefix number with spaces
    while (OutString.length() < LengthNumber)
        OutString = " " + OutString;

    OutString = Suffix + OutString;

    // Prefix entire string with spaces
    NeuronStringSpacePrefixer(Vec, OutString, LengthString);
}

void Setup()
{
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char* glsl_version = "#version 450";
    ImGui_ImplOpenGL3_Init(glsl_version);

    io.Fonts->AddFontFromFileTTF("./res/fonts/Inconsolata-Medium.ttf", Config_::GUI::Font_Size);

    ImGui::StyleColorsDark();

    if (RenderAll)
    {
        RenderOne = false;
        RenderNone = false;
        RenderFit = false;
        RenderAllMatches();
    }
    else if (RenderFit)
    {
        RenderOne = false;
        RenderNone = false;
        RenderFitMatches();
    }
    else if (RenderOne)
    {
        RenderNone = false;
        RenderBestMatch();
    }
    else
    {
        RenderNone = true;
        RenderNoMatches();
    }

    GL::Timer = std::chrono::steady_clock::now();

    int TextLength = 25;

    for (int i = 0; i < SENSORS_EDGE_DISTANCE_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Edge ", 360 / SENSORS_EDGE_DISTANCE_COUNT * i, 3, TextLength);
    for (int i = 0; i < SENSORS_EDGE_DISTANCE_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Versed Edge ", 360 / SENSORS_EDGE_DISTANCE_COUNT * i, 3, TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Velocity X", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Velocity Y", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Versed Velocity X", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Versed Velocity Y", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Angular Velocity", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Versed Angular Velocity", TextLength);
    for (int i = 0; i < SENSORS_EXTERNAL_FORCE_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Acceleration ", 360 / SENSORS_EXTERNAL_FORCE_COUNT * i, 3, TextLength);
    for (int i = 0; i < SENSORS_EXTERNAL_FORCE_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Versed Acceleration ", 360 / SENSORS_EXTERNAL_FORCE_COUNT * i, 3, TextLength);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < SENSORS_ENGINE_ANGLE_COUNT; j++)
            NeuronStringAdder(NeuronInputString, "Engine " + std::to_string(i + 1) + " Angle ", 360 / SENSORS_ENGINE_ANGLE_COUNT * j, 3, TextLength);
        for (int j = 0; j < SENSORS_ENGINE_ANGLE_COUNT; j++)
            NeuronStringAdder(NeuronInputString, "Engine " + std::to_string(i + 1) + " Versed Angle ", 360 / SENSORS_ENGINE_ANGLE_COUNT * j, 3, TextLength);
    }
    for (int i = 0; i < SENSORS_OPPONENT_ANGLE_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Opponent Angle ", 360 / SENSORS_OPPONENT_ANGLE_COUNT * i, 3, TextLength);
    for (int i = 0; i < SENSORS_OPPONENT_ANGLE_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Versed Opponent Angle ", 360 / SENSORS_OPPONENT_ANGLE_COUNT * i, 3, TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Opponent Distance", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Versed Opponent Distance", TextLength);
    for (int i = 0; i < SENSORS_BULLET_ANGLE_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Bullet Angle ", 360 / SENSORS_BULLET_ANGLE_COUNT * i, 3, TextLength);
    for (int i = 0; i < SENSORS_BULLET_ANGLE_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Versed Bullet Angle ", 360 / SENSORS_BULLET_ANGLE_COUNT * i, 3, TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Bullet Distance", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Versed Bullet Distance", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Craft Angle", TextLength);
    NeuronStringSpacePrefixer(NeuronInputString, "Versed Craft Angle", TextLength);
    for (int i = 0; i < SENSORS_MEMORY_COUNT; i++)
        NeuronStringAdder(NeuronInputString, "Memory ", i + 1, 2, TextLength);
    for (int i = 0; i < SENSORS_BIAS_NEURON_AMOUNT; i++)
        NeuronStringAdder(NeuronInputString, "Bias ", i + 1, 1, TextLength);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 3; j++)
            NeuronOutputString.push_back("Engine " + std::to_string(i + 1) + " Rotate Command " + std::to_string(j));
        NeuronOutputString.push_back("Engine " + std::to_string(i + 1) + " Brake");
    }

    for (int i = 0; i < 3; i++)
        NeuronOutputString.push_back("Cannon Rotate Command " + std::to_string(i));
    NeuronOutputString.push_back("Cannon Brake Command");

    NeuronOutputString.push_back("Cannon Fire Command");

    for (int i = 0; i < 4; i++)
        NeuronOutputString.push_back("Engine " + std::to_string(i + 1) + " Thrust Command");

    for (int i = 0; i < SENSORS_MEMORY_COUNT; i++)
        NeuronOutputString.push_back("Memory " + std::to_string(i + 1));
}

void ResetStep() {
    StepNumber = 0;
}

void IncrementMatch() {
    MatchNumber++;
}

void ResetMatch() {
    MatchNumber = 0;
}

void RoundEnd()
{
    RoundNumber++;

    float ScoreCumulative[CRAFT_COUNT];
    cudaCheck(cudaMemcpy(&ScoreCumulative, Crafts->ScoreCumulative, CRAFT_COUNT * sizeof(float), cudaMemcpyDeviceToHost));

    HighScoreCumulative = 0;
    for (int i = 0; i < CRAFT_COUNT; i++) {
        if (ScoreCumulative[i] > HighScoreCumulative)
        {
            HighScoreCumulative = ScoreCumulative[i];
            IndexHighScore = i;
        }
    }

    HighScoreCumulative /= 2.f * 2.f;  // Find average

    if (HighScoreCumulative > HighScoreCumulativeAllTime) {
        HighScoreCumulativeAllTime = HighScoreCumulative;
    }

    HighScoreCumulativeVec.push_back(HighScoreCumulative);
    HighScoreCumulativeVecReverse.push_back(0.f);
    for (int i = 0; i < HighScoreCumulativeVec.size(); i++) {
        HighScoreCumulativeVecReverse[i] = HighScoreCumulativeVec[HighScoreCumulativeVec.size() - 1 - i];
    }

    if (HighScoreCumulativeVec.size() < 250) {
        ProgressDataWidth = GL::ScreenWidth - 15.f;
    }
    else {
        ProgressDataWidth = (float)(HighScoreCumulativeVec.size()) / 250.f * (GL::ScreenWidth - 15.f);
    }

    char Title[64];
    sprintf(Title, "Rnd %d, HS %1.0f", RoundNumber, HighScoreCumulative);
    glfwSetWindowTitle(window, Title);
}

void RoundEnd2()
{
    if (RoundNumber % 2000 == 0)
        SaveFlagEndRound = true;

    if (SaveFlagEndRound)
    {
        // Top
        SaveTopBinary(SaveCount);
        SaveFlagEndRound = false;
    }
}

void AddSpaces(std::string& Output, float Input)
{
    Output = std::to_string(Input);

    if (Input > 0.f)
        Output.insert(0, " ");
    if (Input < 10.f && Input > -10.f)
        Output.insert(0, " ");
}

void StateBar(bool LeftSide, state* d_State, float AngleStart)
{
    char Title[16];

    if (LeftSide)
        sprintf(Title, "Trainee");
    else
        sprintf(Title, "Opponent");

    ImGui::Begin(Title, &ShowStateBar, WindowFlags | ImGuiWindowFlags_HorizontalScrollbar);

    // TODO: Test opponent craft index
    if (LeftSide)
        CopyState<<<1, 1>>>(Crafts, d_State, 0);
    else
        CopyState<<<1, 1>>>(Crafts, d_State, 0 + CRAFT_COUNT);
    cudaCheck(cudaDeviceSynchronize());

    state h_State;
    cudaCheck(cudaMemcpy(&h_State, d_State, sizeof(state), cudaMemcpyDeviceToHost));
    cudaCheck(cudaDeviceSynchronize());

    if (ImGui::CollapsingHeader("Physical State", ImGuiTreeNodeFlags_DefaultOpen)) {
        //TODO: Align spaces

        char GenericCharArray[64];
        sprintf(GenericCharArray, "Score:                  %7.2f", h_State.ScoreBullet + h_State.ScoreTime + h_State.ScoreDistance + h_State.ScoreFuelEfficiency);
        ImGui::Text("%s", GenericCharArray);

        sprintf(GenericCharArray, "Score Cumulative:       %7.2f", h_State.ScoreCumulative);
        ImGui::Text("%s", GenericCharArray);

        std::string GenericString;

        AddSpaces(GenericString, AngleStart);
        sprintf(GenericCharArray, "Starting Angle:        %s", GenericString.c_str());
        ImGui::Text("%s", GenericCharArray);

        std::string GenericString2;

        AddSpaces(GenericString, h_State.PositionX);
        AddSpaces(GenericString2, h_State.PositionY);
        sprintf(GenericCharArray, "Position X:     %s  Y: %s", GenericString.c_str(), GenericString2.c_str());
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString, h_State.VelocityX);
        AddSpaces(GenericString2, h_State.VelocityY);
        sprintf(GenericCharArray, "Velocity X:     %s  Y: %s", GenericString.c_str(), GenericString2.c_str());
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString, h_State.AccelerationX);
        AddSpaces(GenericString2, h_State.AccelerationY);
        sprintf(GenericCharArray, "Acceleration X: %s  Y: %s", GenericString.c_str(), GenericString2.c_str());
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString, h_State.Angle);
        sprintf(GenericCharArray, "Angle:                 %s", GenericString.c_str());
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString, h_State.AngularVelocity);
        sprintf(GenericCharArray, "Angular Velocity:      %s", GenericString.c_str());
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString, h_State.AngularAcceleration);
        sprintf(GenericCharArray, "Angular Acceleration:  %s", GenericString.c_str());
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString, h_State.CannonAngle);
        sprintf(GenericCharArray, "Cannon Angle:          %s", GenericString.c_str());
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString, h_State.CannonCommandAngle);
        sprintf(GenericCharArray, "Command Cannon Angle:  %s", GenericString.c_str());
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString, h_State.CannonStrength);
        sprintf(GenericCharArray, "Cannon Rot Strength:   %s", GenericString.c_str());
        ImGui::Text("%s", GenericCharArray);

        sprintf(GenericCharArray, "Engine:           1       2       3       4");
        ImGui::Text("%s", GenericCharArray);

        sprintf(GenericCharArray, "Angle:          %7.2f %7.2f %7.2f %7.2f", h_State.EngineAngle[0], h_State.EngineAngle[1], h_State.EngineAngle[2], h_State.EngineAngle[3]);
        ImGui::Text("%s", GenericCharArray);

        sprintf(GenericCharArray, "Angle Vel:      %7.2f %7.2f %7.2f %7.2f", h_State.EngineAngularVelocity[0], h_State.EngineAngularVelocity[1], h_State.EngineAngularVelocity[2], h_State.EngineAngularVelocity[3]);
        ImGui::Text("%s", GenericCharArray);

        sprintf(GenericCharArray, "Angle Acc:      %7.2f %7.2f %7.2f %7.2f", h_State.EngineAngularAcceleration[0], h_State.EngineAngularAcceleration[1], h_State.EngineAngularAcceleration[2], h_State.EngineAngularAcceleration[3]);
        ImGui::Text("%s", GenericCharArray);

        sprintf(GenericCharArray, "Thrust Norm:    %7.2f %7.2f %7.2f %7.2f", h_State.EngineThrustNormalized[0], h_State.EngineThrustNormalized[1], h_State.EngineThrustNormalized[2], h_State.EngineThrustNormalized[3]);
        ImGui::Text("%s", GenericCharArray);

        /* std::string GenericString3;
        std::string GenericString4;

        AddSpaces(GenericString,  h_State.EngineThrustNormalized[0]);
        AddSpaces(GenericString2, h_State.EngineThrustNormalized[1]);
        AddSpaces(GenericString3, h_State.EngineThrustNormalized[2]);
        AddSpaces(GenericString4, h_State.EngineThrustNormalized[3]);

        sprintf(GenericCharArray, "Thrust:        %s    %s    %s   %s", GenericString, GenericString2, GenericString3, GenericString4);
        ImGui::Text("%s", GenericCharArray);

        AddSpaces(GenericString,  h_State.EngineAngle[0]);
        AddSpaces(GenericString2, h_State.EngineAngle[1]);
        AddSpaces(GenericString3, h_State.EngineAngle[2]);
        AddSpaces(GenericString4, h_State.EngineAngle[3]);

        sprintf(GenericCharArray, "Angle:        %s    %s    %s   %s", GenericString, GenericString2, GenericString3, GenericString4);
        ImGui::Text("%s", GenericCharArray);*/
    }

    if (ImGui::CollapsingHeader("Neural Network", ImGuiTreeNodeFlags_DefaultOpen)) {
        char GenericString[256];
        sprintf(GenericString, "                                Input ");
        for (int i = 0; i < LAYER_AMOUNT_HIDDEN; i++) {
            std::strcat(GenericString, "      ");
        }
        std::strcat(GenericString, "Output");
        ImGui::Text("%s", GenericString);

        for (int i = 0; i < LAYER_SIZE_INPUT || i < LAYER_SIZE_HIDDEN || i < LAYER_SIZE_OUTPUT; i++) {
            // TODO: Move numbering to GUI setup function
            if (i < 9)
                sprintf(GenericString, "00%d: ", i + 1);
            else if (i < 99)
                sprintf(GenericString, "0%d: ", i + 1);
            else
                sprintf(GenericString, "%d: ", i + 1);

            char NeuronValue[64];

            if (i < LAYER_SIZE_INPUT) {
                if (h_State.Neuron[i] < 0.f)
                    sprintf(NeuronValue, "%s: %1.2f", NeuronInputString[i].c_str(), h_State.Neuron[i]);
                else
                    sprintf(NeuronValue, "%s:  %1.2f", NeuronInputString[i].c_str(), h_State.Neuron[i]);
                strcat(GenericString, NeuronValue);
            }
            else {
                sprintf(NeuronValue, "      ");
                strcat(GenericString, NeuronValue);
            }

            for (int j = 0; j < LAYER_AMOUNT_HIDDEN; j++) {
                if (i < LAYER_SIZE_HIDDEN) {
                    sprintf(NeuronValue, " %5.2f", h_State.Neuron[LAYER_SIZE_INPUT + LAYER_SIZE_HIDDEN * j + i]);
                    strcat(GenericString, NeuronValue);
                }
                else {
                    sprintf(NeuronValue, "      ");
                    strcat(GenericString, NeuronValue);
                }
            }

            if (i < LAYER_SIZE_OUTPUT) {
                if (h_State.Neuron[i + OUTPUT_LAYER_NEURON_BEGIN_INDEX] < 0.f)
                    sprintf(NeuronValue, "  %1.2f: %s", h_State.Neuron[i + OUTPUT_LAYER_NEURON_BEGIN_INDEX], NeuronOutputString[i].c_str());
                else
                    sprintf(NeuronValue, "   %1.2f: %s", h_State.Neuron[i + OUTPUT_LAYER_NEURON_BEGIN_INDEX], NeuronOutputString[i].c_str());
                strcat(GenericString, NeuronValue);
            }

            ImGui::Text("%s", GenericString);
        }
    }

    ImGui::End();
}

void Run(int OpponentID, int PositionNumber, float AngleStart, bool MatchOver)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    /*if (show_demo_window)
    {
        ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
        ImGui::ShowDemoWindow(&show_demo_window);
    }*/

    if (ShowStateBar) {
        state* d_State;
        cudaCheck(cudaMalloc(&d_State, sizeof(state)));
        cudaCheck(cudaDeviceSynchronize());

        // Left
        ImGui::SetNextWindowPos(ImVec2(0, ProgressHeight + MenuHeight), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(StateBarWidth, GL::ScreenHeight - ProgressHeight - MenuHeight), ImGuiCond_Always);

        StateBar(true, d_State, AngleStart);

        // Right
        ImGui::SetNextWindowPos(ImVec2(GL::ScreenWidth - SideBarWidth - StateBarWidth, ProgressHeight + MenuHeight), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(StateBarWidth, GL::ScreenHeight - ProgressHeight - MenuHeight), ImGuiCond_Always);

        StateBar(false, d_State, AngleStart);

        cudaCheck(cudaFree(d_State));
        cudaCheck(cudaDeviceSynchronize());
    }

    if (ShowSideBar) {
        ImGui::SetNextWindowPos(ImVec2(GL::ScreenWidth - SideBarWidth, ProgressHeight + MenuHeight), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(SideBarWidth, GL::ScreenHeight - ProgressHeight - MenuHeight), ImGuiCond_Always);

        ImGui::Begin("SideBar", &ShowSideBar, WindowFlags);

        if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // TODO: Add option for render none
            bool RenderAllWasFalseLastFrame = !RenderAll;
            bool RenderFitWasFalseLastFrame = !RenderFit;
            bool RenderOneFalseLastFrame = !RenderOne;
            bool RenderNoneWasFalseLastFrame = !RenderNone;

            ImGui::Checkbox("Render All", &RenderAll);
            ImGui::Checkbox("Render Fit", &RenderFit);
            ImGui::Checkbox("Render Best from Last Round", &RenderOne);
            ImGui::Checkbox("Render None", &RenderNone);

            if (!RenderAll && !RenderFit && !RenderOne && !RenderNone)
                RenderNone = true;

            if (RenderAllWasFalseLastFrame && RenderAll) {
                RenderOne = false;
                RenderNone = false;
                RenderFit = false;

                RenderAllMatches();
            }
            if (RenderFitWasFalseLastFrame && RenderFit) {
                RenderOne = false;
                RenderNone = false;
                RenderAll = false;

                RenderFitMatches();
            }
            else if (RenderOneFalseLastFrame && RenderOne) {
                RenderAll = false;
                RenderNone = false;
                RenderFit = false;

                RenderBestMatch();
            }
            else if (RenderNoneWasFalseLastFrame && RenderNone) {
                RenderAll = false;
                RenderOne = false;
                RenderFit = false;

                RenderNoMatches();
            }
        }

        if (ImGui::CollapsingHeader("Progress", ImGuiTreeNodeFlags_DefaultOpen)) {
            char GenericString[64];
            sprintf(GenericString, "Runtime: %s", ApplicationRuntime().c_str());
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Round: %d", RoundNumber);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Match: %d", MatchNumber % (2 * 2) + 1);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Current High Score:  %1.0f", HighScoreCumulative);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "All-Time High Score: %1.0f", HighScoreCumulativeAllTime);
            ImGui::Text("%s", GenericString);

            int OpponentRankRangeLast = OpponentRankRange;
            int one = 1;
            ImGui::PushItemWidth(144);
            ImGui::InputScalar("Opponent Rank Range", ImGuiDataType_S32, &OpponentRankRange, &one, NULL, "%d");

            if (OpponentRankRange < OpponentRankRangeLast) {
            if (OpponentRankRange < 1) {
                    OpponentRankRange = 1;
            }
                else {
                    OpponentRankRange += 1;
                    OpponentRankRange /= 2;
                }
            }
            else if (OpponentRankRange > OpponentRankRangeLast) {
                if (OpponentRankRange > FIT_COUNT) {
                    OpponentRankRange = FIT_COUNT;
                }
                else {
                    OpponentRankRange -= 1;
                    OpponentRankRange *= 2;
                }
            }

            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 100);

            sprintf(GenericString, "%d/%d", StepNumber, h_Config->TimeStepLimit);
            float IterationProgressRatio = float(StepNumber) / h_Config->TimeStepLimit;
            ImGui::ProgressBar(IterationProgressRatio, ImVec2(0.f, 0.f), GenericString);
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("Match Progress");

            sprintf(GenericString, "%d/%d", MatchNumber % (MATCHES_PER_ROUND) + 1, MATCHES_PER_ROUND);
            float RoundProgressRatio = float(MatchNumber % (MATCHES_PER_ROUND)) / (MATCHES_PER_ROUND) + IterationProgressRatio / (MATCHES_PER_ROUND);
            ImGui::ProgressBar(RoundProgressRatio, ImVec2(0.f, 0.f), GenericString);
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("Round Progress");

            int SaveAmountLast = SaveCount;
            int One1 = 1;
            ImGui::PushItemWidth(120);
            ImGui::InputScalar("Crafts to Save", ImGuiDataType_S32, &SaveCount, &One1, NULL, "%d");

            if (SaveCount < SaveAmountLast) {
                if (SaveCount < 1) {
                    SaveCount = 1;
                }
                else {
                    SaveCount += 1;
                    SaveCount /= 2;
                }
            }
            else if (SaveCount > SaveAmountLast) {
                SaveCount -= 1;
                SaveCount *= 2;

                if (SaveCount > FIT_COUNT)
                    SaveCount = FIT_COUNT;
            }

            // Check for save button press
            if (SaveFlagEndRound) {
                SaveFlag = ImGui::Button("Save Pending", ImVec2(120.f, 20.f));
            }
            else {
                SaveFlag = ImGui::Button("Save", ImVec2(70.f, 20.f));
            }

            if (SaveFlag) {
                SaveFlagEndRound = true;
                SaveFlag = false;
            }

            if (ImGui::Button("Load Binary")) {
                Pause = true;
                ImGuiFileDialog::Instance()->OpenDialog("Choose craft file to load", "Choose File", ".craft", ".");
            }

            if (ImGuiFileDialog::Instance()->Display("Choose craft file to load")) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                    std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();

                    std::cout << "filePathName: " << filePathName << ", filePath: " << filePath << std::endl;
                    LoadTopBinary(filePathName);
                    exit_round = true;
                }
                // close dialog
                ImGuiFileDialog::Instance()->Close();
                Pause = false;
                // ImGuiFileDialog::Instance()->CloseDialog("ChooseFileDlgKey");
            }
        }

        if (ImGui::CollapsingHeader("Environment Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            char GenericString[64];
            sprintf(GenericString, "Craft Count: %d \tFit Count: %d", CRAFT_COUNT, FIT_COUNT);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Time Limit: %3.0f Seconds", h_Config->TimeLimitMatch);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Physics Time Step: 1/%d Seconds", FRAMERATE_PHYSICS);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Policy  Time Step: 1/%d Seconds", FRAMERATE_NN);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Craft Weight: %4.0f N\tEngine Max Thrust: %4.0f N", CRAFT_MASS * 9.8f, THRUST_MAX);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Bullet Damage: %1.0f", h_Config->BulletDamage);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Neuron Count: %d \tInput: %d \tHidden: %d x %d\tOutput: %d", NEURON_AMOUNT, LAYER_SIZE_INPUT, LAYER_SIZE_HIDDEN, LAYER_AMOUNT_HIDDEN, LAYER_SIZE_OUTPUT);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Weight Count: %d", WEIGHT_AMOUNT);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Total GPU Mem Size: %lu MB", (sizeof(CraftState) + sizeof(MatchState) + sizeof(temp) + sizeof(GraphicsObjectPointer)) / 1024 / 1024);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Weight Array Size: %lu MB", sizeof(float) * WEIGHT_AMOUNT * CRAFT_COUNT / 1024 / 1024);
            ImGui::Text("%s", GenericString);

            sprintf(GenericString, "Startup Time: %4.2f Seconds", TimerStartup);
            ImGui::Text("%s", GenericString);
        }

        if (ImGui::CollapsingHeader("Mutation Parameters")) {
            float MutationFlipChanceLast = h_Config->MutationFlipChance;
            float MutationScaleChanceLast = h_Config->MutationScaleChance;
            float MutationAmountLast = h_Config->MutationScale;
            float MutationSlideChanceLast = h_Config->MutationSlideChance;
            float MutationSigmaLast = h_Config->MutationSigma;

            ImGui::InputFloat("Mutation Flip Chance", &h_Config->MutationFlipChance, 0.001f, 0.01f);
            ImGui::InputFloat("Mutation Scale Chance", &h_Config->MutationScaleChance, 0.001f, 0.01f);
            ImGui::InputFloat("Mutation Scale", &h_Config->MutationScale, 0.001f, 0.01f);
            ImGui::InputFloat("Mutation Slide Chance", &h_Config->MutationSlideChance, 0.001f, 0.01f);
            ImGui::InputFloat("Mutation Sigma", &h_Config->MutationSigma, 0.001f, 0.01f);
            ImGui::Separator();
            ImGui::InputFloat("Max Weight Magnitude", &h_Config->WeightMax, 0.1f, 1.0f);

            if (MutationFlipChanceLast != h_Config->MutationFlipChance || MutationScaleChanceLast != h_Config->MutationScaleChance || MutationAmountLast != h_Config->MutationScale || MutationSlideChanceLast != h_Config->MutationSlideChance || MutationSigmaLast != h_Config->MutationSigma) {
                MutationChangePending = true;
            }

            bool Apply = ImGui::Button("Apply", ImVec2(70.f, 20.f));
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);

            // TODO: Change button colors
            if (MutationChangePending) {
                ImGui::Button("Pending", ImVec2(70.f, 20.f));
            }
            else {
                ImGui::Button("Applied", ImVec2(70.f, 20.f));
            }

            if (Apply) {
                SyncConfigArray();
                MutationChangePending = false;
            }
        }

        if (ImGui::CollapsingHeader("Speed", ImGuiTreeNodeFlags_DefaultOpen)) {
            int TimeSpeedLast = h_Config->TimeSpeed;
            int one = 1;
            ImGui::PushItemWidth(144);
            ImGui::InputScalar("Time Speed", ImGuiDataType_S32, &h_Config->TimeSpeed, &one, NULL, "%d");

            if (h_Config->TimeSpeed < TimeSpeedLast) {
                if (h_Config->TimeSpeed < 1) {
                    h_Config->TimeSpeed = 1;
                }
                else {
                    h_Config->TimeSpeed += 1;
                    h_Config->TimeSpeed /= 2;

                    PerformanceChangePending = true;
                }
            }
            else if (h_Config->TimeSpeed > TimeSpeedLast) {
                if (h_Config->TimeSpeed > 2048) {
                    h_Config->TimeSpeed = 2048;
                }
                else {
                    h_Config->TimeSpeed -= 1;
                    h_Config->TimeSpeed *= 2;

                    PerformanceChangePending = true;
                }
            }

            bool Apply = ImGui::Button("Apply", ImVec2(70.f, 20.f));
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            if (PerformanceChangePending) {
                ImGui::Button("Pending", ImVec2(70.f, 20.f));
            }
            else {
                ImGui::Button("Applied", ImVec2(70.f, 20.f));
            }

            if (Apply) {
                SyncConfigArray();
                PerformanceChangePending = false;
            }

            if (!SimulationSpeedToggle) {
                if (SimulateFastFlag) {
                    SimulationSpeedToggle = ImGui::Button("Simulate Real-Time", ImVec2(140.f + ImGui::GetStyle().ItemInnerSpacing.x, 20.f));\
                }
                else
                {
                    SimulationSpeedToggle = ImGui::Button("Simulate Fast", ImVec2(140.f + ImGui::GetStyle().ItemInnerSpacing.x, 20.f));
                    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);

                    int TimeSpeedFastLast = h_Config->TimeSpeedFast;
                    int One = 1;
                    ImGui::PushItemWidth(100);
                    ImGui::InputScalar("Time Speed Fast", ImGuiDataType_S32, &h_Config->TimeSpeedFast, &One, NULL, "%d");

                    if (h_Config->TimeSpeedFast < TimeSpeedFastLast)
                    {
                        if (h_Config->TimeSpeedFast < 2)
                            h_Config->TimeSpeedFast = 2;
                        else
                        {
                            h_Config->TimeSpeedFast += 1;
                            h_Config->TimeSpeedFast /= 2;
                        }
                    }
                    else if (h_Config->TimeSpeedFast > TimeSpeedFastLast)
                    {
                        h_Config->TimeSpeedFast -= 1;
                        h_Config->TimeSpeedFast *= 2;

                        if (h_Config->TimeSpeedFast > h_Config->TimeSpeedFastDefault)
                            h_Config->TimeSpeedFast = h_Config->TimeSpeedFastDefault;
                    }
                }
            }

            if (SimulationSpeedToggle) {
                SimulateFastFlag = !SimulateFastFlag;

                if (SimulateFastFlag)
                {
                    h_Config->TimeSpeed = h_Config->TimeSpeedFast;

                    RenderNone = true;
                    RenderAll = false;
                    RenderOne = false;
                    RenderFit = false;

                    SyncConfigArray();
                    RenderNoMatches();
                }
                else
                {
                    h_Config->TimeSpeed = 1;
                    RenderAll = false;
                    RenderOne = true;
                    RenderNone = false;
                    RenderFit = false;

                    SyncConfigArray();
                    RenderBestMatch();
                }

                SimulationSpeedToggle = false;
            }

            bool PauseFlag = false;
            if (Pause)
                PauseFlag = ImGui::Button("Continue", ImVec2(140.f + ImGui::GetStyle().ItemInnerSpacing.x, 20.f));
            else
                PauseFlag = ImGui::Button("Pause", ImVec2(140.f + ImGui::GetStyle().ItemInnerSpacing.x, 20.f));

            if (PauseFlag)
                Pause = !Pause;

            ImGui::Separator();

            float FrameTimeMcs = (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - GL::Timer).count();
            GL::Timer = std::chrono::steady_clock::now();

            char FrameRateString[32];
            float FrameRate = 1000000.f / FrameTimeMcs;
            sprintf(FrameRateString, "%2.1f/%d.0", FrameRate, int(FRAMES_PER_SECOND));
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 100);
            ImGui::ProgressBar(FrameRate / FRAMES_PER_SECOND, ImVec2(0.f, 0.f), FrameRateString);
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("Frame Rate");
        }

        if (ShowSideBar)
            SideBarWidth = SideBarWidthDefault;
        else
            SideBarWidth = 0;

        ImGui::End();
    }

    if (ShowProgress)
    {
        ImGui::SetNextWindowPos(ImVec2(0, MenuHeight), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(GL::ScreenWidth, ProgressHeight), ImGuiCond_Always);

        //ImGuiWindowFlags_HorizontalScrollbar
        ImGui::Begin("High Scores Per Round", &ShowProgress, WindowFlags | ImGuiWindowFlags_AlwaysHorizontalScrollbar);
        float* HighScoreCumulativeVecReverseData = static_cast<float*>(HighScoreCumulativeVecReverse.data());
        ImGui::PlotLines("", HighScoreCumulativeVecReverseData, HighScoreCumulativeVecReverse.size() + 1, 0, "", 0.f,
            (float)HighScoreCumulativeAllTime, ImVec2(ProgressDataWidth, ProgressHeight - 55.f), 4);

        ImGui::End();

        if (ShowProgress)
            ProgressHeight = ProgressHeightDefault;
        else
            ProgressHeight = 0;
    }

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            ImGui::MenuItem("Nothing");
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window"))
        {
            ImGui::Checkbox("Show Side Bar", &ShowSideBar);
            ImGui::Checkbox("Show Progress Chart", &ShowProgress);
            ImGui::Checkbox("Show State Bar", &ShowStateBar);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (!Pause)
        StepNumber += h_Config->IterationsPerCall;
}

void Shutdown()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

// TODO: Add List of all best IDs from each round

} // End namespace GUI