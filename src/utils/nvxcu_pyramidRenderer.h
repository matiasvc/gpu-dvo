//
// Created by matiasvc on 05.03.19.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_NVXCU_PYRAMIDRENDERER_H
#define DIRECT_IMAGE_ALIGNMENT_NVXCU_PYRAMIDRENDERER_H

#include <vector>
#include <NVX/nvxcu.h>

#include "nvxcu_utils.h"
#include "nvxcu_imageRenderer.h"

class nvxcuPyramidRenderer
{
public:
	nvxcuPyramidRenderer(int32_t width, int32_t height, nvxcu_df_image_e type, unsigned int levels, float scale=NVXCU_SCALE_PYRAMID_HALF);
	
	unsigned int getRenderTexture(unsigned int level) const;
	int32_t getWidth(unsigned int level) const;
	int32_t getHeight(unsigned int level) const;
	unsigned int getNumberOfLevels() const;
	
	void drawPyramid(const nvxcu_pitch_linear_pyramid_t& pyramid);
	
private:
	const unsigned int m_levels;
	std::vector<nvxcuImageRenderer> m_imageRenderers;
};


#endif //DIRECT_IMAGE_ALIGNMENT_NVXCU_PYRAMIDRENDERER_H
