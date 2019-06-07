//
// Created by matiasvc on 05.03.19.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_NVXCU_IMAGERENDERER_H
#define DIRECT_IMAGE_ALIGNMENT_NVXCU_IMAGERENDERER_H

#include <stdint.h>
#include <memory>
#include <driver_types.h>

#include "nvxcu_utils.h"
#include "../gl/Shader.h"

class nvxcuImageRenderer
{
public:
	nvxcuImageRenderer(int32_t width, int32_t height, nvxcu_df_image_e type, bool jetColorMap = false);
	
	unsigned int getRenderTexture() const;
	int32_t getWidth() const;
	int32_t getHeight() const;
	
	void initializeKeypointRendering(uint32_t capacity, nvxcu_array_item_type_e keypointType);
	void drawImage(const nvxcu_pitch_linear_image_t& image);
	void drawKeypointArray(const nvxcu_plain_array_t& array, int32_t nKeypoints);
	
private:
	static unsigned int m_VBO;
	static unsigned int m_VAO;
	static unsigned int m_EBO;
	
	const int32_t m_width;
	const int32_t m_height;
	const nvxcu_df_image_e m_imageType;
	nvxcu_array_item_type_e m_keypointType;
	
	uint32_t m_verticesCapacity;
	
	unsigned int m_imageTexture;
	unsigned int m_renderTexture;
	unsigned int m_frameBuffer;
	unsigned int m_vertexBufferObject;
	
	unsigned int m_pointVAO;
	
	std::unique_ptr<Shader> m_imageShaderPtr;
	std::unique_ptr<Shader> m_verticesShaderPtr;
	
	cudaGraphicsResource_t m_textureGraphicsResource;
	cudaGraphicsResource_t m_verticesGraphicsResource;
	cudaStream_t m_stream;
	
};

#endif //DIRECT_IMAGE_ALIGNMENT_NVXCU_IMAGERENDERER_H
