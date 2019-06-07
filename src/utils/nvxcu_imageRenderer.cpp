//
// Created by matiasvc on 05.03.19.
//

#include "nvxcu_imageRenderer.h"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <iostream>

#include "../gl/glDebug.h"
#include "nvxcu_debug.h"
#include "../cuda/cudaUtils.h"

#include "../../shaders/shaders.h"


unsigned int nvxcuImageRenderer::m_VBO = 0;
unsigned int nvxcuImageRenderer::m_VAO = 0;
unsigned int nvxcuImageRenderer::m_EBO = 0;

nvxcuImageRenderer::nvxcuImageRenderer(int32_t width, int32_t height, nvxcu_df_image_e type, bool jetColorMap)
: m_width{width}, m_height{height}, m_imageType{type}, m_keypointType{(nvxcu_array_item_type_e)0}, m_verticesCapacity{0}, m_imageTexture{0}, m_renderTexture{0}, m_frameBuffer{0}, m_vertexBufferObject{0}, m_pointVAO{0},
  m_textureGraphicsResource{nullptr}, m_verticesGraphicsResource{nullptr}, m_stream{nullptr}
{
	if (!nvxcuImageRenderer::m_VAO) // Setup VAO for the image, this should only happen once.
	{
		const float vertices[] = {
				// positions        // texture coords
				1.0f, -1.0f, 0.0f, 1.0f, 1.0f, // bottom right
				1.0f,  1.0f, 0.0f, 1.0f, 0.0f, // top right
				-1.0f,  1.0f, 0.0f, 0.0f, 0.0f, // top left
				-1.0f, -1.0f, 0.0f, 0.0f, 1.0f  // bottom left
		};
		const unsigned int indices[] = {
				0, 1, 3, // first triangle
				1, 2, 3  // second triangle
		};
		
		glGenVertexArrays(1, &m_VAO);
		glGenBuffers(1, &m_VBO);
		glGenBuffers(1, &m_EBO);
		
		glBindVertexArray(m_VAO);
		
		glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
		
		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		// texture coord attribute
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);
		
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
	
	
	
	glGenTextures(1, &m_imageTexture);
	glBindTexture(GL_TEXTURE_2D, m_imageTexture);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	
	float borderColor[] = { 1.0f, 1.0f, 0.0f, 1.0f };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	if (type == NVXCU_DF_IMAGE_RGB)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr); glCheckError();
		m_imageShaderPtr = std::make_unique<Shader>(image_vs_str, rgb_image_fs_str);
	}
	else if(type == NVXCU_DF_IMAGE_U8)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr); glCheckError();
		if (jetColorMap)
		{
			m_imageShaderPtr = std::make_unique<Shader>(image_vs_str, gray_image_jet_fs_str);
		}
		else
		{
			m_imageShaderPtr = std::make_unique<Shader>(image_vs_str, gray_image_fs_str);
		}
		
	}
	else if(type == NVXCU_DF_IMAGE_U16)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, nullptr); glCheckError();
		m_imageShaderPtr = std::make_unique<Shader>(image_vs_str, grayU16_image_fs_str);
	}
	else if(type == NVXCU_DF_IMAGE_S16)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, nullptr); glCheckError();
		m_imageShaderPtr = std::make_unique<Shader>(image_vs_str, gradientU16_image_fs_str);
	}
	else
	{
		throw std::invalid_argument("Unsupported image type");
	}
	
	
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage(&m_textureGraphicsResource, m_imageTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard | cudaGraphicsRegisterFlagsSurfaceLoadStore) );
	CUDA_SAFE_CALL( cudaStreamCreate(&m_stream) );
	
	glGenFramebuffers(1, &m_frameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_frameBuffer);
	
	glGenTextures(1, &m_renderTexture);
	glBindTexture(GL_TEXTURE_2D, m_renderTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_renderTexture, 0);
	
	GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, drawBuffers);
	
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cerr << "ERROR: Framebuffer\n";
	}
	
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


unsigned int nvxcuImageRenderer::getRenderTexture() const
{
	return m_renderTexture;
}

int32_t nvxcuImageRenderer::getWidth() const
{
	return m_width;
}

int32_t nvxcuImageRenderer::getHeight() const
{
	return m_height;
}

void nvxcuImageRenderer::initializeKeypointRendering(uint32_t capacity, nvxcu_array_item_type_e keypointType)
{
	m_keypointType = keypointType;
	glGenVertexArrays(1, &m_pointVAO);
	glGenBuffers(1, &m_vertexBufferObject);
	
	glBindVertexArray(m_pointVAO);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObject);
	
	switch (keypointType)
	{
		case NVXCU_TYPE_KEYPOINT:
		{
			const unsigned int size = capacity * 2 * sizeof(float);
			glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
			
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), nullptr);
			glEnableVertexAttribArray(0);
			
			glCheckError();
			
			m_verticesShaderPtr = std::make_unique<Shader>(point2d_vs_str, point2d_fs_str);
		} break;
		default:
		{
			std::cerr << "ERROR! Unsupported keypoint type: " << keypointType << '\n';
		} break;
	}
	
	
	
	
	glBindVertexArray(0);
	
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&m_verticesGraphicsResource, m_vertexBufferObject, cudaGraphicsMapFlagsWriteDiscard) );
	
	m_verticesCapacity = capacity;
}

void nvxcuImageRenderer::drawImage(const nvxcu_pitch_linear_image_t& image)
{
	assert(image.base.format == m_imageType);
	assert(image.base.width == m_width);
	assert(image.base.height == m_height);
	
	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &m_textureGraphicsResource, m_stream) );
	cudaArray_t cudaArr = nullptr;
	CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray(&cudaArr, m_textureGraphicsResource, 0, 0) );
	
	if (m_imageType == NVXCU_DF_IMAGE_RGB)
	{
		CUDA_SAFE_CALL( cudaScatterRGBToGLArray(cudaArr, image.planes[0].dev_ptr, image.planes[0].pitch_in_bytes,
		                                        m_width, m_height, m_stream) );
	}
	else if (m_imageType == NVXCU_DF_IMAGE_U16)
	{
		CUDA_SAFE_CALL( cudaScatterShortToGLArray(cudaArr, image.planes[0].dev_ptr, image.planes[0].pitch_in_bytes,
		                                          m_width, m_height, m_stream) );
	}
	else if (m_imageType == NVXCU_DF_IMAGE_S16)
	{
		CUDA_SAFE_CALL( cudaScatterSignedShortToGLArray(cudaArr, image.planes[0].dev_ptr, image.planes[0].pitch_in_bytes,
		                                                m_width, m_height, m_stream) );
	}
	else if(m_imageType == NVXCU_DF_IMAGE_U8)
	{
		CUDA_SAFE_CALL( cudaMemcpy2DToArrayAsync(cudaArr, 0, 0,
		                                         image.planes[0].dev_ptr, (size_t)image.planes[0].pitch_in_bytes,
		                                         sizeof(uint8_t) * image.base.width, image.base.height,
		                                         cudaMemcpyDeviceToDevice, m_stream)
		);
	}
	else
	{
		throw std::invalid_argument("Unsupported image type");
	}
	
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &m_textureGraphicsResource, m_stream) );
	CUDA_SAFE_CALL( cudaStreamSynchronize(m_stream) );
	
	glBindFramebuffer(GL_FRAMEBUFFER, m_frameBuffer);
	glViewport(0, 0, m_width, m_height);
	glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_imageTexture);
	
	m_imageShaderPtr->use();
	glBindVertexArray(m_VAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
	
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}
