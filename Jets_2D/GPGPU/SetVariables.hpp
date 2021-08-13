#pragma once

// Standard Library
#include <cmath>

// Project Headers
#include "Jets_2D/Config.hpp"
#include "Jets_2D/ErrorCheck.hpp"
#include "Jets_2D/GPGPU/GPSetup.hpp"
#include "Jets_2D/GPGPU/GPErrorCheck.hpp"

// Turns on rendering of all matches
void RenderAllMatches();

// Turns on rendering of fit matches
void RenderFitMatches();

// Turns on rendering of match of best craft
void RenderBestMatch();

// Turns off all match rendering
void RenderNoMatches();

// Call everytime h_Config is modifed
void SyncConfigArray();
