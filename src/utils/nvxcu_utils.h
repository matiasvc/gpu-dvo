//
// Created by matiasvc on 22.02.19.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_NVXCU_UTILS_H
#define DIRECT_IMAGE_ALIGNMENT_NVXCU_UTILS_H

#include "nvxcu_debug.h"

#include <utility>
#include <NVX/nvxcu.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <vector>


inline constexpr size_t nPlanesFromType(const nvxcu_df_image_e type)
{
	switch (type)
	{
		case NVXCU_DF_IMAGE_RGB:
			return 1;
		case NVXCU_DF_IMAGE_RGBX:
			return 1;
		case NVXCU_DF_IMAGE_UYVY:
			return 1;
		case NVXCU_DF_IMAGE_YUYV:
			return 1;
		case NVXCU_DF_IMAGE_IYUV:
			return 3;
		case NVXCU_DF_IMAGE_YUV4:
			return 3;
		case NVXCU_DF_IMAGE_U8:
			return 1;
		case NVXCU_DF_IMAGE_U16:
			return 1;
		case NVXCU_DF_IMAGE_S16:
			return 1;
		case NVXCU_DF_IMAGE_U32:
			return 1;
		case NVXCU_DF_IMAGE_S32:
			return 1;
		case NVXCU_DF_IMAGE_F32:
			return 1;
		case NVXCU_DF_IMAGE_2F32:
			return 1;
		case NVXCU_DF_IMAGE_2S16:
			return 1;
		case NVXCU_DF_IMAGE_RGB16:
			return 1;
		default:
			throw std::invalid_argument("Invalid type");
	}
}

inline constexpr size_t texelSizeFromType(const nvxcu_df_image_e type)
{
	switch (type)
	{
		case NVXCU_DF_IMAGE_RGB:
			return 3 * sizeof(uint8_t);
		case NVXCU_DF_IMAGE_RGBX:
			return 4 * sizeof(uint8_t);
		case NVXCU_DF_IMAGE_UYVY:
			return 4 * sizeof(uint8_t);
		case NVXCU_DF_IMAGE_YUYV:
			return 4 * sizeof(uint8_t);
		case NVXCU_DF_IMAGE_IYUV:
			return 1 * sizeof(uint8_t);
		case NVXCU_DF_IMAGE_YUV4:
			return 1 * sizeof(uint8_t);
		case NVXCU_DF_IMAGE_U8:
			return 1 * sizeof(uint8_t);
		case NVXCU_DF_IMAGE_U16:
			return 1 * sizeof(uint16_t);
		case NVXCU_DF_IMAGE_S16:
			return 1 * sizeof(int16_t);
		case NVXCU_DF_IMAGE_U32:
			return 1 * sizeof(uint32_t);
		case NVXCU_DF_IMAGE_S32:
			return 1 * sizeof(int32_t);
		case NVXCU_DF_IMAGE_F32:
			return 1 * sizeof(float);
		case NVXCU_DF_IMAGE_2F32:
			return 2 * sizeof(float);
		case NVXCU_DF_IMAGE_2S16:
			return 2 * sizeof(int16_t);
		case NVXCU_DF_IMAGE_RGB16:
			return 3 * sizeof(uint16_t);
		default:
			throw std::invalid_argument("Invalid type");
	}
}

struct nvxcu_pitch_linear_image_container
{
	explicit nvxcu_pitch_linear_image_container(const nvxcu_pitch_linear_image_t& image)
		: m_contained{image},
		  m_nPlanes{nPlanesFromType(image.base.format)},
		  m_texelSize{texelSizeFromType(image.base.format)}
	{ }
	
	~nvxcu_pitch_linear_image_container()
	{
		for (int plane = 0; plane < m_nPlanes; ++plane)
		{
			cudaFree(m_contained.planes[plane].dev_ptr);
		}
	}
	
	const nvxcu_pitch_linear_image_t m_contained;
	const size_t m_nPlanes;
	const size_t m_texelSize;
};

struct nvxcu_plain_array_container
{
	explicit nvxcu_plain_array_container(const nvxcu_plain_array_t& array)
		: m_contained{array}
	{ }
	
	~nvxcu_plain_array_container()
	{
		cudaFree(m_contained.num_items_dev_ptr);
		cudaFree(m_contained.dev_ptr);
	}
	
	const nvxcu_plain_array_t m_contained;
};

struct nvxcu_linear_pyramid_container
{
	explicit nvxcu_linear_pyramid_container(const nvxcu_pitch_linear_pyramid_t& pyramid)
		: m_contained{pyramid}
	{ }
	
	~nvxcu_linear_pyramid_container()
	{
		for (int level = 0; level < m_contained.base.num_levels; ++level)
		{
			size_t nPlanes = nPlanesFromType(m_contained.levels[level].base.format);
			for (int plane = 0; plane < nPlanes; ++plane)
			{
				cudaFree(m_contained.levels[level].planes[plane].dev_ptr);
			}
		}
		free(m_contained.levels);
	}
	
	nvxcu_pitch_linear_pyramid_t m_contained;
};

struct nvxcu_tmp_buf_container
{
	explicit nvxcu_tmp_buf_container(const nvxcu_tmp_buf_t& tmp_buf)
		: m_contained{tmp_buf}
	{ }
	
	~nvxcu_tmp_buf_container()
	{
		if (m_contained.dev_ptr) { cudaFree(m_contained.dev_ptr); }
		if (m_contained.host_ptr) { free(m_contained.host_ptr); }
	}
	
	const nvxcu_tmp_buf_t m_contained;
};

inline nvxcu_pitch_linear_image_t createImage(uint32_t width, uint32_t height, const nvxcu_df_image_e type)
{
	nvxcu_pitch_linear_image_t image;
	image.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
	image.base.format = type;
	image.base.width = width;
	image.base.height = height;
	
	auto planes = nPlanesFromType(type);
	auto elementSize = texelSizeFromType(type);
	
	for (int plane = 0; plane < planes; ++plane)
	{
		void *dev_ptr = nullptr;
		size_t pitch = 0;
		CUDA_SAFE_CALL( cudaMallocPitch(&dev_ptr, &pitch, width * elementSize, height) );
		image.planes[plane].dev_ptr = dev_ptr;
		image.planes[plane].pitch_in_bytes = (uint32_t)pitch;
	}
	
	return image;
}

template <typename T>
inline nvxcu_plain_array_t createPlainArray(uint32_t capacity, nvxcu_array_item_type_e type)
{
	void *dev_ptr = nullptr;
	cudaMalloc(&dev_ptr, capacity * sizeof(T));
	
	uint32_t *num_items_dev_ptr = nullptr;
	cudaMalloc((void**)&num_items_dev_ptr, sizeof(uint32_t));
	
	nvxcu_plain_array_t array;
	array.base.array_type = NVXCU_PLAIN_ARRAY;
	array.base.item_type = type;
	array.base.capacity = capacity;
	array.dev_ptr = dev_ptr;
	array.num_items_dev_ptr = num_items_dev_ptr;
	
	return array;
}

inline nvxcu_pitch_linear_pyramid_t createPyramidHalfScale(size_t width, size_t height, nvxcu_df_image_e imageType, size_t numLevels)
{
	nvxcu_pitch_linear_pyramid_t pyramid;
	
	pyramid.base.pyramid_type = NVXCU_PITCH_LINEAR_PYRAMID;
	pyramid.base.num_levels = (uint32_t)numLevels;
	pyramid.base.scale = NVXCU_SCALE_PYRAMID_HALF;
	
	pyramid.levels = (nvxcu_pitch_linear_image_t*)calloc(numLevels, sizeof(nvxcu_pitch_linear_image_t));
	
	auto currentWidth = (uint32_t)width;
	auto currentHeight = (uint32_t)height;
	float currentScale = 1.0f;
	
	for (int i = 0; i < numLevels; ++i)
	{
		pyramid.levels[i] = createImage(currentWidth, currentHeight, imageType);
		
		// Next level dimensions
		currentScale *= pyramid.base.scale;
		currentWidth = (uint32_t)std::ceil((float)width * currentScale);
		currentHeight = (uint32_t)std::ceil((float)height * currentScale);
	}
	
	return pyramid;
}

inline void calculateGradientPyramids(const nvxcu_pitch_linear_pyramid_t& imagePyramid, const nvxcu_pitch_linear_pyramid_t& gradXPyramid, const nvxcu_pitch_linear_pyramid_t& gradYPyramid,
                                      const nvxcu_border_t& border, const nvxcu_stream_exec_target_t& exec_target)
{
	assert(imagePyramid.levels[0].base.format == NVXCU_DF_IMAGE_U8);
	assert(gradXPyramid.levels[0].base.format == NVXCU_DF_IMAGE_S16);
	assert(gradYPyramid.levels[0].base.format == NVXCU_DF_IMAGE_S16);
	
	for (int level = 0; level < imagePyramid.base.num_levels; ++level)
	{
		NVXCU_SAFE_CALL( nvxcuScharr3x3(&imagePyramid.levels[level].base, &gradXPyramid.levels[level].base, &gradYPyramid.levels[level].base, &border, &exec_target.base) );
	}
}

inline nvxcu_tmp_buf_size_t totalBufSize(std::vector<nvxcu_tmp_buf_size_t> bufSizes)
{
	nvxcu_tmp_buf_size_t totalSize = {0, 0};
	for (auto& bufSize: bufSizes)
	{
		totalSize.host_buf_size = std::max(totalSize.host_buf_size, bufSize.host_buf_size);
		totalSize.dev_buf_size = std::max(totalSize.dev_buf_size, bufSize.dev_buf_size);
	}
	return totalSize;
}

inline nvxcu_tmp_buf_t createTmpBuf(std::vector<nvxcu_tmp_buf_size_t> bufSizes)
{
	nvxcu_tmp_buf_t buf{};
	nvxcu_tmp_buf_size_t bufSize = totalBufSize(std::move(bufSizes));
	
	if (bufSize.host_buf_size > 0)
	{
		cudaMallocHost(&buf.host_ptr, bufSize.host_buf_size);
	}
	
	if (bufSize.dev_buf_size > 0)
	{
		cudaMalloc(&buf.dev_ptr, bufSize.dev_buf_size);
	}
	
	return buf;
}

inline nvxcu_tmp_buf_t createTmpBuf(nvxcu_tmp_buf_size_t bufSize)
{
	nvxcu_tmp_buf_t buf{};
	
	cudaMallocHost(&buf.host_ptr, bufSize.host_buf_size);
	cudaMalloc(&buf.dev_ptr, bufSize.dev_buf_size);
	
	return buf;
}


#endif //DIRECT_IMAGE_ALIGNMENT_NVXCU_UTILS_H
