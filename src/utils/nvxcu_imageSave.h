//
// Created by matiasvc on 05.04.19.
//

#ifndef GPU_STEREO_DSO_NVXCU_IMAGESAVE_H
#define GPU_STEREO_DSO_NVXCU_IMAGESAVE_H

#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <stdint.h>
#include <NVX/nvxcu.h>

#include "nvxcu_debug.h"

inline void saveNVXCUImage(const nvxcu_pitch_linear_image_t& img, const std::string& filename, const uint32_t targetWidth, const uint32_t targetHeight)
{
	assert(img.base.format == NVXCU_DF_IMAGE_U8);
	
	nvxcu_pitch_linear_image_t resizedImage;
	{
		resizedImage.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
		resizedImage.base.format = NVXCU_DF_IMAGE_U8;
		resizedImage.base.width = targetWidth;
		resizedImage.base.height = targetHeight;
		
		void *dev_ptr = nullptr;
		size_t pitch = 0;
		CUDA_SAFE_CALL( cudaMallocPitch(&dev_ptr, &pitch, targetWidth * sizeof(uint8_t), targetHeight) );
		resizedImage.planes[0].dev_ptr = dev_ptr;
		resizedImage.planes[0].pitch_in_bytes = (uint32_t)pitch;
	}
	
	nvxcu_border_t border;
	border.mode = NVXCU_BORDER_MODE_REPLICATE;
	
	nvxcu_stream_exec_target_t exec_target;
	exec_target.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
	cudaStreamCreate(&exec_target.stream);
	cudaGetDeviceProperties(&exec_target.dev_prop, 0);
	
	nvxcuScaleImage(&img.base, &resizedImage.base, NVXCU_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, &border, &exec_target.base);
	
	cudaStreamSynchronize(exec_target.stream);
	cudaStreamDestroy(exec_target.stream);
	
	const size_t dataSize = resizedImage.planes[0].pitch_in_bytes*resizedImage.base.height;
	void* h_data = malloc(dataSize);
	CUDA_SAFE_CALL( cudaMemcpy(h_data, resizedImage.planes[0].dev_ptr, dataSize, cudaMemcpyDeviceToHost) );
	
	stbi_write_png(filename.c_str(), resizedImage.base.width, resizedImage.base.height, 1, h_data, resizedImage.planes[0].pitch_in_bytes);
	
	free(h_data);
	CUDA_SAFE_CALL( cudaFree(resizedImage.planes[0].dev_ptr) );
}

#endif //GPU_STEREO_DSO_NVXCU_IMAGESAVE_H
