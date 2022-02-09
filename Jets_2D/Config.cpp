// File Header
#include "Jets_2D/Config.hpp"

namespace Config_
{
    // Default rendering
    bool RenderAllDefault = false;
    bool RenderOneDefault = true;
    bool RenderFitDefault = false;
    bool RenderNoneDefault = false;      // Setting this true defaults fast simulation

    int LayerSizeArray[LAYER_AMOUNT] = LAYER_ARRAY;
}

namespace Config_
{   namespace GUI
    {
        float Font_Size = 12.f;
    }
}