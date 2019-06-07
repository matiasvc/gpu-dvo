#include "imageUtils.h"

#include <cassert>
#include <iostream>
#include <NVX/nvxcu.h>
#include <stdint.h>
#include <Eigen/Core>
#include <cassert>

#include "../primitives/DVOPoint.h"

__device__ __constant__  float gaussianKernel[5][5] =
{
	{1.0,  4.0,  6.0,  4.0, 1.0},
	{4.0, 16.0, 24.0, 16.0, 4.0},
	{6.0, 24.0, 36.0, 24.0, 6.0},
	{4.0, 16.0, 24.0, 16.0, 4.0},
	{1.0,  4.0,  6.0,  4.0, 1.0}
};



__global__
void calculateDepthPyramidKernel(void* sourceImage, int32_t sourceImagePitch,
                                 uint32_t sourceImageWidth, uint32_t sourceImageHeight,
                                 void* targetImage, int32_t targetImagePitch,
                                 uint32_t targetImageWidth, uint32_t targetImageHeight)
{
	const int targetU = blockIdx.x*blockDim.x + threadIdx.x;
	const int targetV = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (targetU >= targetImageWidth or targetV >= targetImageHeight) { return; }
	
	const int sourceU = targetU*2;
	const int sourceV = targetV*2;
	
	float valueSum = 0.0f;
	float kernelSum = 0.0f;
	
	const int kernelSize = 5;
	const int kernelHalfSize = 2;
	
	char* sourceBytePointer = (char*)sourceImage;
	
	for (int v = 0; v < kernelSize; ++v)
	{
		for (int u = 0; u < kernelSize; ++u)
		{
			int localSourceU = sourceU - kernelHalfSize + u;
			int localSourceV = sourceV - kernelHalfSize + v;
			
			localSourceU = max(localSourceU, 0);
			localSourceU = min(localSourceU, sourceImageWidth-1);
			
			localSourceV = max(localSourceV, 0);
			localSourceV = min(localSourceV, sourceImageHeight-1);
			
			char* localBytePointer = &sourceBytePointer[localSourceV*sourceImagePitch + localSourceU*sizeof(uint16_t)];
			
			uint16_t sourceValue = *((uint16_t*)localBytePointer);
			
			if (sourceValue != 0)
			{
				const float kernel_value = gaussianKernel[v][u];
				
				kernelSum += kernel_value;
				valueSum += kernel_value * (float)sourceValue;
			}
		}
	}
	
	const float targetValueFloat = valueSum / kernelSum;
	const uint16_t targetValue = (uint16_t)nearbyint(targetValueFloat);
	
	const char* targetBytePointer = &((char*)targetImage)[targetV*targetImagePitch + targetU*sizeof(uint16_t)];
	uint16_t* targetPointer = (uint16_t*)targetBytePointer;
	
	*targetPointer = targetValue;
}

cudaError_t calculateDepthPyramid(const nvxcu_pitch_linear_pyramid_t& depthPyramid, cudaStream_t stream)
{
	assert(depthPyramid.levels[0].base.format == NVXCU_DF_IMAGE_U16);
	assert(depthPyramid.base.scale == NVXCU_SCALE_PYRAMID_HALF);
	assert(depthPyramid.base.num_levels > 1);
	
	cudaError_t err;
	
	dim3 blockDim(32, 32, 1);
	
	for (int layer = 1; layer < depthPyramid.base.num_levels; ++layer)
	{
		const auto sourceWidth = depthPyramid.levels[layer-1].base.width;
		const auto sourceHeight = depthPyramid.levels[layer-1].base.height;
		
		const auto targetWidth = depthPyramid.levels[layer].base.width;
		const auto targetHeight = depthPyramid.levels[layer].base.height;
		
		dim3 gridDim(targetWidth/blockDim.x + (targetWidth % blockDim.x == 0 ? 0 : 1),
		             targetHeight/blockDim.y + (targetHeight % blockDim.y == 0 ? 0 : 1),
		             1);
		
		calculateDepthPyramidKernel<<<gridDim, blockDim, 0, stream>>>(depthPyramid.levels[layer-1].planes[0].dev_ptr, depthPyramid.levels[layer-1].planes[0].pitch_in_bytes,
		                                                              sourceWidth, sourceHeight,
		                                                              depthPyramid.levels[layer].planes[0].dev_ptr, depthPyramid.levels[layer].planes[0].pitch_in_bytes,
		                                                              targetWidth, targetHeight);
	}
	
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		std::cerr << "KERNEL ERROR DEPTH PYRAMID: " << cudaGetErrorString(err) << std::endl;
	}
	
	
	return err;
}

__global__
void convertU16ToU8(void* sourceImage, int32_t sourceImagePitch,
                    void* targetImage, int32_t targetImagePitch,
                    const uint32_t imageWidth, const uint32_t imageHeight)
{
	const int u = blockIdx.x*blockDim.x + threadIdx.x;
	const int v = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (u >= imageWidth or v >= imageHeight) { return; }
	
	char* sourceBytePointer = (char*)sourceImage;
	sourceBytePointer += v*sourceImagePitch + u*sizeof(uint16_t);
	uint16_t* sourcePointer = (uint16_t*)sourceBytePointer;
	
	uint16_t sourceValue = *sourcePointer;
	uint8_t targetValue = (uint8_t)nearbyint(((double)sourceValue)*(255.0/65535.0));
	
	char* targetBytePointer = (char*)targetImage;
	targetBytePointer += v*targetImagePitch + u*sizeof(uint8_t);
	uint8_t* targetPointer = (uint8_t*)targetBytePointer;
	
	*targetPointer = targetValue;
}

cudaError_t convertU16ToU8(const nvxcu_pitch_linear_image_t& source, const nvxcu_pitch_linear_image_t& target, cudaStream_t stream)
{
	assert(source.base.format == NVXCU_DF_IMAGE_U16);
	assert(target.base.format == NVXCU_DF_IMAGE_U8);
	
	assert(source.base.width == target.base.width);
	assert(source.base.height == target.base.height);
	
	const auto width = source.base.width;
	const auto height = source.base.height;
	
	cudaError_t err;
	dim3 blockDim(32, 32, 1);
	dim3 gridDim(width/blockDim.x + (width % blockDim.x == 0 ? 0 : 1),
	             height/blockDim.y + (height % blockDim.y == 0 ? 0 : 1),
	             1);
	
	convertU16ToU8<<<gridDim, blockDim, 0, stream>>>(source.planes[0].dev_ptr, source.planes[0].pitch_in_bytes,
	                                                 target.planes[0].dev_ptr, target.planes[0].pitch_in_bytes,
	                                                 width, height);
	
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		std::cerr << "KERNEL ERROR U16 to U8: " << cudaGetErrorString(err) << std::endl;
	}
	
	
	return err;
}

__global__
void projectDepthPixelsKernel(void* grayImage, uint32_t grayImagePitch,
                              void* depthImage, uint32_t depthImagePitch,
                              const uint32_t imageWidth, const uint32_t imageHeight,
                              DVOPoint* pointArray, uint32_t* numItemsPtr, IntrinsicParameters intrinsicParameters)
{
	const int u = blockIdx.x*blockDim.x + threadIdx.x;
	const int v = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (u >= imageWidth or v >= imageHeight) { return; }
	
	const uint16_t* depthValuePtr = (uint16_t*)(((uint8_t*)depthImage) + v*depthImagePitch + u*sizeof(uint16_t));
	const uint16_t depthValue = * depthValuePtr;
	
	if (depthValue == 0) { return; }
	
	const uint32_t arrayIndex = atomicInc(numItemsPtr, 0xFFFFFFFF);
	
	const uint8_t* grayValuePtr = ((uint8_t*)grayImage) + v*grayImagePitch + u*sizeof(uint8_t);
	const uint8_t grayValue = *grayValuePtr;
	
	pointArray[arrayIndex].intensity = grayValue;
	
	const double z = static_cast<double>(depthValue) / 5000.0;
	const double x = (u - intrinsicParameters.cx) * z / intrinsicParameters.fx;
	const double y = (v - intrinsicParameters.cy) * z / intrinsicParameters.fy;
	
	pointArray[arrayIndex].pos = Eigen::Vector3d(x, y, z);
}

void projectDepthPixels(const nvxcu_pitch_linear_image_t& grayImage, const nvxcu_pitch_linear_image_t& depthImage,
                        const nvxcu_plain_array_t& pointArray, const IntrinsicParameters intrinsicParameters, cudaStream_t stream)
{
	assert(grayImage.base.width == depthImage.base.width);
	assert(grayImage.base.height == depthImage.base.height);
	
	cudaMemset(pointArray.num_items_dev_ptr, 0, sizeof(uint32_t));
	
	const uint32_t imageWidth = grayImage.base.width;
	const uint32_t imageHeight = grayImage.base.height;
	
	dim3 blockDim(32, 32, 1);
	
	dim3 gridDim(imageWidth / blockDim.x + (imageWidth % blockDim.x == 0  ? 0 : 1),
	             imageHeight / blockDim.y + (imageHeight % blockDim.y == 0  ? 0 : 1),
	             1);
	
	projectDepthPixelsKernel<<<gridDim, blockDim, 0, stream>>>(grayImage.planes[0].dev_ptr, grayImage.planes[0].pitch_in_bytes,
	                                                           depthImage.planes[0].dev_ptr, depthImage.planes[0].pitch_in_bytes,
	                                                           imageWidth, imageHeight, (DVOPoint*)pointArray.dev_ptr, pointArray.num_items_dev_ptr, intrinsicParameters);
}
