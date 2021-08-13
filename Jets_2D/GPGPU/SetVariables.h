#pragma once

// Standard Library
#include <cmath>

// Project Headers
#include "Jets_2D/Config.h"
#include "Jets_2D/ErrorCheck.h"
#include "Jets_2D/GPGPU/GPSetup.h"
#include "Jets_2D/GPGPU/GPErrorCheck.h"

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
