#pragma once

#include <NVX/nvxcu.h>
#include <sophus/se3.hpp>
#include <Eigen/Dense>
#include "primitives/InstrinsicParameters.h"
#include <array>

template <int LEVELS>
struct AlignmentStatistics
{
	std::array<uint32_t, LEVELS> iterationPerLevel;
	long elapsedMicroseconds;
};

struct AlignmentSettings
{
	uint32_t maxIterations;
	double motionPrior;
	double xiConvergenceLength;
	double initialStepLength;
	double stepLengthReductionFactor;
	double minStepLength;
};

struct AlignImageBuffers
{
	double* d_tmpSumValue;
	double* d_tmpSumArray;
	uint32_t* d_nActivePoints;
	double* d_J;
	double* d_Jw;
	double* d_residual;
	double* d_weights;
	double* d_A;
	double* d_b;
};

nvxcu_tmp_buf_size_t alignImagesBuffSize(uint32_t baseImageWidth, uint32_t baseImageHeight);

AlignImageBuffers createAlignImageBuffers(void* tmpBuffer, uint32_t baseImageWidth, uint32_t baseImageHeight);

Eigen::Matrix<double, 6, 1> alignImages(const nvxcu_pitch_linear_pyramid_t& currentImagePyramid, const nvxcu_pitch_linear_pyramid_t& currentImageGradXPyramid, const nvxcu_pitch_linear_pyramid_t& currentImageGradYPyramid,
                 const nvxcu_pitch_linear_pyramid_t& previousImagePyramid, const nvxcu_pitch_linear_pyramid_t& previousDepthPyramid, const nvxcu_plain_array_t& pointArray, const IntrinsicParameters& intrinsicParameters, const Eigen::Matrix<double, 6, 1>& initialXi,
                 const AlignImageBuffers& alignImageBuffers, const AlignmentSettings& alignmentSettings, AlignmentStatistics<5>& alignmentStatistics);
