#include "cudaUtils.h"

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <cmath>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <NVX/nvxcu.h>

surface<void, cudaSurfaceType2D> surfaceWrite;


__global__
void scatterShortToGLArrayKernel(const void* src, int32_t spitch, size_t textureWidth, size_t textureHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= textureWidth or y >= textureHeight) { return; }
	
	char* valuePtr = ((char*)src) + x*sizeof(uint16_t) + y*spitch;
	
	uint16_t value = *((uint16_t*)valuePtr);
	
	surf2Dwrite<uint16_t>(value, surfaceWrite, x*sizeof(uint16_t), y, cudaBoundaryModeTrap);
}

cudaError_t cudaScatterShortToGLArray(cudaArray_t dst, const void* src, int32_t spitch, int32_t textureWidth, int32_t textureHeight, cudaStream_t stream)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim(textureWidth/blockDim.x + (textureWidth % blockDim.x == 0 ? 0 : 1),
	             textureHeight/blockDim.y + (textureHeight % blockDim.y == 0 ? 0 : 1),
	             1);
	
	cudaError_t err;
	
	err = cudaBindSurfaceToArray(surfaceWrite, dst);
	if (err != cudaSuccess)
	{
		std::cerr << "ERROR BINDING SURFACE SHORT: " << cudaGetErrorString(err) << std::endl;
		return err;
	}
	
	scatterShortToGLArrayKernel<<<gridDim, blockDim, 0, stream>>>(src, spitch, textureWidth, textureHeight);
	
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		std::cerr << "KERNEL ERROR SHORT: " << cudaGetErrorString(err) << std::endl;
	}
	
	return err;
}

__global__
void scatterSignedShortToGLArrayKernel(const void* src, int32_t spitch, size_t textureWidth, size_t textureHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= textureWidth || y >= textureHeight) { return; }
	
	char* valuePtr = ((char*)src) + x*sizeof(int16_t) + y*spitch;
	
	const int32_t uint16MaxValue = 32768;
	
	int16_t value = *((int16_t*)valuePtr);
	int32_t value32 = (int32_t)value + uint16MaxValue;
	uint16_t value16 = (uint16_t)value32;
	
	
	surf2Dwrite<uint16_t>(value16, surfaceWrite, x*sizeof(uint16_t), y, cudaBoundaryModeTrap);
}

cudaError_t cudaScatterSignedShortToGLArray(cudaArray_t dst, const void* src, int32_t spitch, int32_t textureWidth, int32_t textureHeight, cudaStream_t stream)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim(textureWidth/blockDim.x + (textureWidth % blockDim.x == 0 ? 0 : 1),
	             textureHeight/blockDim.y + (textureHeight % blockDim.y == 0 ? 0 : 1),
	             1);
	
	cudaError_t err;
	
	err = cudaBindSurfaceToArray(surfaceWrite, dst);
	if (err != cudaSuccess)
	{
		std::cerr << "ERROR BINDING SURFACE SIGNED SHORT: " << cudaGetErrorString(err) << std::endl;
		return err;
	}
	
	scatterSignedShortToGLArrayKernel<<<gridDim, blockDim, 0, stream>>>(src, spitch, textureWidth, textureHeight);
	
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		std::cerr << "KERNEL ERROR SIGNED SHORT: " << cudaGetErrorString(err) << std::endl;
	}
	
	return err;
}

__global__
void scatterRGBToGLArrayKernel(const void* src, int32_t spitch, size_t textureWidth, size_t textureHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= textureWidth || y >= textureHeight) { return; }
	
	unsigned char* valuePtr = ((unsigned char*)src) + x*sizeof(uchar3) + y*spitch;
	
	unsigned char* rPtr = valuePtr;
	unsigned char* gPtr = valuePtr + 1;
	unsigned char* bPtr = valuePtr + 2;
	
	unsigned char rValue = *(rPtr);
	unsigned char gValue = *(gPtr);
	unsigned char bValue = *(bPtr);
	
	surf2Dwrite<unsigned char>(rValue, surfaceWrite, x*sizeof(char4)+0, y, cudaBoundaryModeTrap);
	surf2Dwrite<unsigned char>(gValue, surfaceWrite, x*sizeof(char4)+1, y, cudaBoundaryModeTrap);
	surf2Dwrite<unsigned char>(bValue, surfaceWrite, x*sizeof(char4)+2, y, cudaBoundaryModeTrap);
}

cudaError_t cudaScatterRGBToGLArray(cudaArray_t dst, const void* src, int32_t spitch, int32_t textureWidth, int32_t textureHeight, cudaStream_t stream)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim(textureWidth/blockDim.x + (textureWidth % blockDim.x == 0 ? 0 : 1),
	             textureHeight/blockDim.y + (textureHeight % blockDim.y == 0 ? 0 : 1),
	             1);
	
	cudaError_t err;
	
	err = cudaBindSurfaceToArray(surfaceWrite, dst);
	if (err != cudaSuccess)
	{
		std::cerr << "ERROR BINDING SURFACE RGB: " << cudaGetErrorString(err) << std::endl;
		return err;
	}
	
	scatterRGBToGLArrayKernel<<<gridDim, blockDim, 0, stream>>>(src, spitch, textureWidth, textureHeight);
	
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		std::cerr << "KERNEL ERROR RGB: " << cudaGetErrorString(err) << std::endl;
	}
	
	return err;
}



