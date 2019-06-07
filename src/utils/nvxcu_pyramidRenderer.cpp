//
// Created by matiasvc on 05.03.19.
//

#include "nvxcu_pyramidRenderer.h"

#include <cassert>

nvxcuPyramidRenderer::nvxcuPyramidRenderer(int32_t width, int32_t height, nvxcu_df_image_e type, unsigned int levels, float scale)
: m_levels{levels}
{
	auto currentWidth = width;
	auto currentHeight = height;
	float currentScale = 1.0;
	
	for (int level = 0; level < m_levels; ++level)
	{
		m_imageRenderers.emplace_back(currentWidth, currentHeight, type);
		
		currentScale *= scale;
		currentWidth = (uint32_t)std::ceil((float)width * currentScale);
		currentHeight = (uint32_t)std::ceil((float)height * currentScale);
	}
}

unsigned int nvxcuPyramidRenderer::getRenderTexture(unsigned int level) const
{
	assert(level < m_levels);
	return m_imageRenderers[level].getRenderTexture();
}

int32_t nvxcuPyramidRenderer::getWidth(unsigned int level) const
{
	assert(level < m_levels);
	return m_imageRenderers[level].getWidth();
}

int32_t nvxcuPyramidRenderer::getHeight(unsigned int level) const
{
	assert(level < m_levels);
	return m_imageRenderers[level].getHeight();
}

unsigned int nvxcuPyramidRenderer::getNumberOfLevels() const
{
	return m_levels;
}

void nvxcuPyramidRenderer::drawPyramid(const nvxcu_pitch_linear_pyramid_t& pyramid)
{
	for (int level = 0; level < m_levels; ++level)
	{
		const auto& levelImage = pyramid.levels[level];
		m_imageRenderers[level].drawImage(levelImage);
	}
}
