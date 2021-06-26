// File Header
#include "Config.h"

namespace Config_
{
	// Default rendering
	bool RenderAllDefault = false;
	bool RenderOneDefault = false;
	bool RenderFitDefault = false;
	bool RenderNoneDefault = true;		// Setting this true defaults fast simulation

	int LayerSizeArray[LAYER_AMOUNT] = LAYER_ARRAY;
}

namespace Config_
{	namespace GUI
	{
		float Font_Size = 12.f;
	}
}