//
// Created by matiasvc on 01.03.19.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_CUDA_UTILS_H
#define DIRECT_IMAGE_ALIGNMENT_CUDA_UTILS_H

#include <driver_types.h>
#include <NVX/nvxcu.h>

#include "../primitives/InstrinsicParameters.h"

cudaError_t cudaScatterShortToGLArray(cudaArray_t dst, const void* src, int32_t spitch, int32_t textureWidth, int32_t textureHeight, cudaStream_t stream);
cudaError_t cudaScatterSignedShortToGLArray(cudaArray_t dst, const void* src, int32_t spitch, int32_t textureWidth, int32_t textureHeight, cudaStream_t stream);
cudaError_t cudaScatterRGBToGLArray(cudaArray_t dst, const void* src, int32_t spitch, int32_t textureWidth, int32_t textureHeight, cudaStream_t stream);

cudaError_t cudaCopyPlainArrayToGLBuffer(void* dstPtr, const nvxcu_plain_array_t& array, uint32_t nVertices, uint32_t imageWidth, uint32_t imageHeight, cudaStream_t stream);

#endif //DIRECT_IMAGE_ALIGNMENT_CUDA_UTILS_H
