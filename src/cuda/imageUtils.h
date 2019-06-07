//
// Created by matiasvc on 26.03.19.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_IMAGEUTILS_H
#define DIRECT_IMAGE_ALIGNMENT_IMAGEUTILS_H

#include <NVX/nvxcu.h>
#include <cuda_runtime.h>
#include "../primitives/InstrinsicParameters.h"

cudaError_t calculateDepthPyramid(const nvxcu_pitch_linear_pyramid_t& depthPyramid, cudaStream_t stream);
cudaError_t convertU16ToU8(const nvxcu_pitch_linear_image_t& source, const nvxcu_pitch_linear_image_t& target, cudaStream_t stream);

void projectDepthPixels(const nvxcu_pitch_linear_image_t& grayImage, const nvxcu_pitch_linear_image_t& depthImage,
                        const nvxcu_plain_array_t& pointArray, IntrinsicParameters intrinsicParameters, cudaStream_t stream);

#endif //DIRECT_IMAGE_ALIGNMENT_IMAGEUTILS_H
