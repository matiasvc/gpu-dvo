#include "imageAlignment.h"

#include <cassert>
#include <cmath>
#include <sophus/se3.hpp>
#include <NVX/nvxcu.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <limits>
#include <inttypes.h>
#include <iomanip>
#include <cublas_v2.h>
#include <chrono>

#include "utils/asserts.h"
#include "utils/nvxcu_debug.h"
#include "utils/nvxcu_utils.h"
#include "utils/nvxcu_imageSave.h"
#include "cuda/imageUtils.h"
#include "primitives/DVOPoint.h"




using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

__global__
void calculateJacobianAndResidual(const Eigen::Matrix3d rotation, const Eigen::Vector3d translation,
                                  const cudaTextureObject_t currentImageTexture,
                                  const cudaTextureObject_t gradXTexture, const cudaTextureObject_t gradYTexture,
                                  const uint32_t imageWidth, const int32_t imageHeight,
                                  const IntrinsicParameters intrinsicParameters,
                                  const DVOPoint* pointArray, const uint32_t nPoints,
                                  double* Jptr, double* residualPtr, uint32_t* nActivePointsPtr)
{
	const uint32_t pointId = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (pointId >= nPoints) { return; }
	
	const DVOPoint point = pointArray[pointId];
	const Eigen::Vector3d projectedPoint = rotation * point.pos + translation;
	
	const double& x = projectedPoint(0);
	const double& y = projectedPoint(1);
	const double& z = projectedPoint(2);
	
	const double& fx = intrinsicParameters.fx;
	const double& fy = intrinsicParameters.fy;
	const double& cx = intrinsicParameters.cx;
	const double& cy = intrinsicParameters.cy;
	
	const double u = fx*x/z + cx;
	const double v = fy*y/z + cy;
	
	if (u >= 0.0 and u < static_cast<double>(imageWidth-1) and v >= 0.0 and v < static_cast<double>(imageHeight-1)) // Is the point inside the image
	{
		const float gradXValue = tex2D<float>(gradXTexture, static_cast<float>(u + 0.5), static_cast<float>(v + 0.5));
		const float gradYValue = tex2D<float>(gradYTexture, static_cast<float>(u + 0.5), static_cast<float>(v + 0.5));

		const double Ix = static_cast<double>(gradXValue);
		const double Iy = static_cast<double>(gradYValue);

		const double z_div = 1.0/z;
		const double z2_div = 1.0/(z*z);

		Jptr[pointId + 0*nPoints] =  (Ix*fx)*z_div;
		Jptr[pointId + 1*nPoints] =  (Iy*fy)*z_div;
		Jptr[pointId + 2*nPoints] = -(Ix*fx*x)*z2_div - (Iy*fy*y)*z2_div;
		Jptr[pointId + 3*nPoints] = -Iy*fy - y*((Ix*fx*x) + (Iy*fy*y))*z2_div;
		Jptr[pointId + 4*nPoints] =  Ix*fx + x*((Ix*fx*x) + (Iy*fy*y))*z2_div;
		Jptr[pointId + 5*nPoints] = (Iy*fy*x)*z_div - (Ix*fx*y)*z_div;
		
		const double currentImageValue = static_cast<double>(tex2D<float>(currentImageTexture, static_cast<float>(u + 0.5), static_cast<float>(v + 0.5)));
		const double previousImageValue = static_cast<double>(point.intensity)/std::numeric_limits<uint8_t>::max();
		const double residual = currentImageValue - previousImageValue;
		residualPtr[pointId] = residual;
		
		atomicInc(nActivePointsPtr, 0xFFFFFFFF); // Count up the number of points that are active in the optimization
	}
	else
	{
		Jptr[pointId + 0*nPoints] = 0.0;
		Jptr[pointId + 1*nPoints] = 0.0;
		Jptr[pointId + 2*nPoints] = 0.0;
		Jptr[pointId + 3*nPoints] = 0.0;
		Jptr[pointId + 4*nPoints] = 0.0;
		Jptr[pointId + 5*nPoints] = 0.0;
		
		residualPtr[pointId] = 0.0;
	}
}

__inline__ __device__
double warpReduceSum(double val)
{
	const unsigned int mask = __activemask();
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val += __shfl_down_sync(mask, val, offset);
	}
	return val;
}

__inline__ __device__
double blockReduceSum(double val)
{
	static __shared__ double shared[32];
	const int lane = threadIdx.x % warpSize;
	const int wid = threadIdx.x / warpSize;
	
	val = warpReduceSum(val); // Each warp performs partial reduction
	
	if (lane == 0) { shared[wid] = val; } // Write reduced value to shared memory
	
	__syncthreads();
	
	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
	
	//Final reduce within first warp
	if (wid == 0) { val = warpReduceSum(val); }
	
	return val;
}

__global__ void deviceReduceSum(double* in, double* out, const int N)
{
	double sum = 0.0;
	const uint32_t arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Reduce multiple elements per thread
	for (int i = arrayIndex; i < N; i += blockDim.x * gridDim.x)
	{
		sum += in[i];
	}
	
	sum = blockReduceSum(sum);
	
	if (threadIdx.x == 0)
	{
		out[blockIdx.x] = sum;
	}
}

__global__ void calculateStdDev(double* residualArray, double* weights, double* residualSum, const uint32_t nArrayElements, uint32_t* nActivePointsPtr)
{
	const uint32_t arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (arrayIndex >= nArrayElements) { return; }
	
	const double mean = *residualSum / (*nActivePointsPtr);
	const double diff = residualArray[arrayIndex] - mean;
	
	weights[arrayIndex] = diff*diff;
}

__global__ void calculateHuberWeights(double* residualArray, double* weightArray, double* variancePtr, const uint32_t nArrayElements, uint32_t* nActivePointsPtr)
{
	const uint32_t arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (arrayIndex >= nArrayElements) { return; }
	
	// Value of 1.345 gives 95% efficiency in cases of gaussian distribution and is commonly used with huber weights
	const double k = 1.345*sqrt((1.0/(*nActivePointsPtr))*(*variancePtr));
	const double absError = abs(residualArray[arrayIndex]);
	
	if (absError > k)
	{
		weightArray[arrayIndex] = k/absError;
	}
	else
	{
		weightArray[arrayIndex] = 1.0;
	}
}

__global__ void calculateWeightedJacobian(double* Jwptr, double* Jptr, double* weightsPtr, const uint32_t nPoints)
{
	const uint32_t pointId = blockIdx.x*blockDim.x + threadIdx.x;
	if (pointId >= nPoints) { return; }
	
	const double weight = weightsPtr[pointId];
	
	Jwptr[pointId + 0*nPoints] = Jptr[pointId + 0*nPoints]*weight;
	Jwptr[pointId + 1*nPoints] = Jptr[pointId + 1*nPoints]*weight;
	Jwptr[pointId + 2*nPoints] = Jptr[pointId + 2*nPoints]*weight;
	Jwptr[pointId + 3*nPoints] = Jptr[pointId + 3*nPoints]*weight;
	Jwptr[pointId + 4*nPoints] = Jptr[pointId + 4*nPoints]*weight;
	Jwptr[pointId + 5*nPoints] = Jptr[pointId + 5*nPoints]*weight;
}

inline size_t toNext8(size_t size)
{
	if (size % 8 == 0)
	{
		return size;
	}
	else
	{
		return ((size/8) + 1)*8;
	}
}

nvxcu_tmp_buf_size_t alignImagesBuffSize(const uint32_t baseImageWidth, const uint32_t baseImageHeight)
{
	nvxcu_tmp_buf_size_t tmpBufSize{
		.dev_buf_size = 0,
		.host_buf_size = 0,
	};
	
	const uint32_t maxNumberOfPoints = baseImageWidth*baseImageHeight;
	
	tmpBufSize.dev_buf_size += toNext8(sizeof(double)); // d_tmpSumValue
	tmpBufSize.dev_buf_size += toNext8(maxNumberOfPoints * sizeof(double)); // d_tmpSumArray
	tmpBufSize.dev_buf_size += toNext8(sizeof(uint32_t)); // d_nActivePoints
	
	tmpBufSize.dev_buf_size += toNext8(maxNumberOfPoints * 6 * sizeof(double)); // d_J
	tmpBufSize.dev_buf_size += toNext8(maxNumberOfPoints * 6 * sizeof(double)); // d_Jw
	tmpBufSize.dev_buf_size += toNext8(maxNumberOfPoints * sizeof(double)); // d_residual
	tmpBufSize.dev_buf_size += toNext8(maxNumberOfPoints * sizeof(double)); // d_weights
	
	tmpBufSize.dev_buf_size += toNext8(6 * 6 * sizeof(double)); // d_A
	tmpBufSize.dev_buf_size += toNext8(6 * sizeof(double)); // d_b
	
	return tmpBufSize;
}



AlignImageBuffers createAlignImageBuffers(void* tmpBuffer, const uint32_t baseImageWidth, const uint32_t baseImageHeight)
{
	const uint32_t maxNumberOfPoints = baseImageWidth*baseImageHeight;
	AlignImageBuffers buffers{};
	
	uint8_t* bytePtr = reinterpret_cast<uint8_t*>(tmpBuffer);
	
	buffers.d_tmpSumValue = reinterpret_cast<double*>(bytePtr);
	bytePtr += toNext8(sizeof(double));
	
	buffers.d_tmpSumArray = reinterpret_cast<double*>(bytePtr);
	bytePtr += toNext8(maxNumberOfPoints * sizeof(double));
	
	buffers.d_nActivePoints = reinterpret_cast<uint32_t*>(bytePtr);
	bytePtr += toNext8(sizeof(uint32_t));
	
	buffers.d_J = reinterpret_cast<double*>(bytePtr);
	bytePtr += toNext8(maxNumberOfPoints * 6 * sizeof(double));
	
	buffers.d_Jw = reinterpret_cast<double*>(bytePtr);
	bytePtr += toNext8(maxNumberOfPoints * 6 * sizeof(double));
	
	buffers.d_residual = reinterpret_cast<double*>(bytePtr);
	bytePtr += toNext8(maxNumberOfPoints * sizeof(double));
	
	buffers.d_weights = reinterpret_cast<double*>(bytePtr);
	bytePtr += toNext8(maxNumberOfPoints * sizeof(double));
	
	buffers.d_A = reinterpret_cast<double*>(bytePtr);
	bytePtr += toNext8(6 * 6 * sizeof(double));
	
	buffers.d_b = reinterpret_cast<double*>(bytePtr);
	//bytePtr += toNext8(6*sizeof(double));
	
	return buffers;
}

Eigen::Matrix<double, 6, 1> alignImages(const nvxcu_pitch_linear_pyramid_t& currentImagePyramid, const nvxcu_pitch_linear_pyramid_t& currentImageGradXPyramid, const nvxcu_pitch_linear_pyramid_t& currentImageGradYPyramid,
	const nvxcu_pitch_linear_pyramid_t& previousImagePyramid, const nvxcu_pitch_linear_pyramid_t& previousDepthPyramid, const nvxcu_plain_array_t& pointArray, const IntrinsicParameters& intrinsicParameters,
	const Eigen::Matrix<double, 6, 1>& initialXi, const AlignImageBuffers& alignImageBuffers, const AlignmentSettings& alignmentSettings, AlignmentStatistics<5>& alignmentStatistics)
{
	assert(all_equal_image_size({currentImagePyramid, currentImageGradXPyramid, currentImageGradYPyramid, previousImagePyramid, previousDepthPyramid}));
	assert(currentImagePyramid.levels[0].base.format == NVXCU_DF_IMAGE_U8);
	assert(currentImageGradXPyramid.levels[0].base.format == NVXCU_DF_IMAGE_S16);
	assert(currentImageGradYPyramid.levels[0].base.format == NVXCU_DF_IMAGE_S16);
	
	auto start = std::chrono::steady_clock::now();
	
	cudaStream_t stream;
	CUDA_SAFE_CALL( cudaStreamCreate(&stream) );
	
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);
	cublasSetStream_v2(cublasHandle, stream);
	
	
	const uint32_t levels = currentImagePyramid.base.num_levels;
	Vector6d xi = initialXi;
	Matrix6d sigma_inv = Matrix6d::Identity()*(1.0/alignmentSettings.motionPrior);
	
	for (int level = levels-1; level >= 0 ; --level)
	{
		const nvxcu_pitch_linear_image_t& currentImage = currentImagePyramid.levels[level];
		const nvxcu_pitch_linear_image_t& currentImageGradX = currentImageGradXPyramid.levels[level];
		const nvxcu_pitch_linear_image_t& currentImageGradY = currentImageGradYPyramid.levels[level];
		
		const nvxcu_pitch_linear_image_t& previousImage = previousImagePyramid.levels[level];
		const nvxcu_pitch_linear_image_t& previousDepth = previousDepthPyramid.levels[level];
		
		const double levelIntrinsicMultiplier = 1.0 / std::exp2(static_cast<double>(level));
		const IntrinsicParameters levelIntrinsics{intrinsicParameters.fx*levelIntrinsicMultiplier, intrinsicParameters.fy*levelIntrinsicMultiplier, intrinsicParameters.cx*levelIntrinsicMultiplier, intrinsicParameters.cy*levelIntrinsicMultiplier};
		
		cudaMemsetAsync(pointArray.num_items_dev_ptr, 0, sizeof(uint32_t), stream); // Reset counter
		projectDepthPixels(previousImage, previousDepth, pointArray, levelIntrinsics, stream); // Populate pointArray
		
		uint32_t nPoints;
		cudaMemcpyAsync(reinterpret_cast<void*>(&nPoints), reinterpret_cast<void*>(pointArray.num_items_dev_ptr), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		
		cudaResourceDesc currentImageDesc{};
		currentImageDesc.resType = cudaResourceTypePitch2D;
		currentImageDesc.res.pitch2D.devPtr = currentImage.planes[0].dev_ptr;
		currentImageDesc.res.pitch2D.desc = cudaCreateChannelDesc(8*sizeof(uint8_t), 0, 0, 0, cudaChannelFormatKindUnsigned);
		currentImageDesc.res.pitch2D.width = currentImage.base.width;
		currentImageDesc.res.pitch2D.height = currentImage.base.height;
		currentImageDesc.res.pitch2D.pitchInBytes = static_cast<size_t>(currentImage.planes[0].pitch_in_bytes);
		
		cudaResourceDesc gradXResourceDesc{};
		gradXResourceDesc.resType = cudaResourceTypePitch2D;
		gradXResourceDesc.res.pitch2D.devPtr = currentImageGradX.planes[0].dev_ptr;
		gradXResourceDesc.res.pitch2D.desc = cudaCreateChannelDesc(8*sizeof(int16_t), 0, 0, 0, cudaChannelFormatKindSigned);
		gradXResourceDesc.res.pitch2D.width = currentImageGradX.base.width;
		gradXResourceDesc.res.pitch2D.height = currentImageGradX.base.height;
		gradXResourceDesc.res.pitch2D.pitchInBytes = static_cast<size_t>(currentImageGradX.planes[0].pitch_in_bytes);
		
		cudaResourceDesc gradYResourceDesc{};
		gradYResourceDesc.resType = cudaResourceTypePitch2D;
		gradYResourceDesc.res.pitch2D.devPtr = currentImageGradY.planes[0].dev_ptr;
		gradYResourceDesc.res.pitch2D.desc = cudaCreateChannelDesc(8*sizeof(int16_t), 0, 0, 0, cudaChannelFormatKindSigned);
		gradYResourceDesc.res.pitch2D.width = currentImageGradY.base.width;
		gradYResourceDesc.res.pitch2D.height = currentImageGradY.base.height;
		gradYResourceDesc.res.pitch2D.pitchInBytes = static_cast<size_t>(currentImageGradY.planes[0].pitch_in_bytes);
		
		cudaTextureDesc textureDesc{};
		textureDesc.addressMode[0] = cudaAddressModeClamp;
		textureDesc.addressMode[1] = cudaAddressModeClamp;
		textureDesc.filterMode = cudaFilterModeLinear;
		textureDesc.readMode = cudaReadModeNormalizedFloat;
		
		cudaTextureObject_t currentImageTextureObject;
		cudaTextureObject_t gradXTextureObject;
		cudaTextureObject_t gradYTextureObject;
		CUDA_SAFE_CALL( cudaCreateTextureObject(&currentImageTextureObject, &currentImageDesc, &textureDesc, nullptr) );
		CUDA_SAFE_CALL( cudaCreateTextureObject(&gradXTextureObject, &gradXResourceDesc, &textureDesc, nullptr) );
		CUDA_SAFE_CALL( cudaCreateTextureObject(&gradYTextureObject, &gradYResourceDesc, &textureDesc, nullptr) );
		
		const dim3 jacobianBlockDim(1024, 1, 1);
		const dim3 jacobianGridDim(nPoints / jacobianBlockDim.x + (nPoints % jacobianBlockDim.x == 0  ? 0 : 1), 1, 1);
		
		const dim3 reductionBlockDim(1024, 1, 1);
		const dim3 reductionGridDim(nPoints / reductionBlockDim.x + (nPoints % reductionBlockDim.x == 0 ? 0 : 1), 1, 1);
		assert(nPoints <= reductionBlockDim.x * reductionGridDim.x);
		
		const dim3 stdDevBlockDim(1024, 1, 1);
		const dim3 stdDebGridDim(nPoints / stdDevBlockDim.x + (nPoints % stdDevBlockDim.x == 0  ? 0 : 1), 1, 1);
		
		const dim3 huberBlockDim(1024, 1, 1);
		const dim3 huberGridDim(nPoints / huberBlockDim.x + (nPoints % huberBlockDim.x == 0  ? 0 : 1), 1, 1);
		
		const dim3 weightedJacobianBlockDim(1024, 1, 1);
		const dim3 weightedJacobianGridDim(nPoints / weightedJacobianBlockDim.x + (nPoints % weightedJacobianBlockDim.x == 0  ? 0 : 1), 1, 1);
		
		double stepLength = alignmentSettings.initialStepLength;
		
		for (int iteration = 0; iteration < alignmentSettings.maxIterations; ++iteration)
		{
			const Sophus::SE3d currentPose = Sophus::SE3d::exp(xi);
			
			calculateJacobianAndResidual<<<jacobianGridDim, jacobianBlockDim, 0, stream>>>(
					currentPose.rotationMatrix(), currentPose.translation(),
					currentImageTextureObject,
					gradXTextureObject, gradYTextureObject,
					currentImageGradX.base.width, currentImageGradX.base.height,
					levelIntrinsics,
					reinterpret_cast<DVOPoint*>(pointArray.dev_ptr), nPoints,
					alignImageBuffers.d_J, alignImageBuffers.d_residual, alignImageBuffers.d_nActivePoints);
			
			
			// Sum residuals
			deviceReduceSum<<<reductionGridDim, reductionBlockDim, 0, stream>>>(alignImageBuffers.d_residual, alignImageBuffers.d_tmpSumArray, nPoints);
			deviceReduceSum<<<1, 1024, 0, stream>>>(alignImageBuffers.d_tmpSumArray, alignImageBuffers.d_tmpSumValue, nPoints); // Store sum in d_tmpSumValue
			
			// Store squared difference from mean in d_weights
			calculateStdDev<<<stdDebGridDim, stdDevBlockDim, 0, stream>>>(alignImageBuffers.d_residual, alignImageBuffers.d_weights, alignImageBuffers.d_tmpSumValue, nPoints, alignImageBuffers.d_nActivePoints);
			
			// Sum squared differences
			deviceReduceSum<<<reductionGridDim, reductionBlockDim, 0, stream>>>(alignImageBuffers.d_weights, alignImageBuffers.d_tmpSumArray, nPoints);
			deviceReduceSum<<<1, 1024, 0, stream>>>(alignImageBuffers.d_tmpSumArray, alignImageBuffers.d_tmpSumValue, nPoints); // Store sum in d_tmpSumValue
			
			calculateHuberWeights<<<huberGridDim, huberBlockDim, 0, stream>>>(alignImageBuffers.d_residual, alignImageBuffers.d_weights, alignImageBuffers.d_tmpSumValue, nPoints, alignImageBuffers.d_nActivePoints);
			
			calculateWeightedJacobian<<<weightedJacobianGridDim, weightedJacobianBlockDim, 0, stream>>>(alignImageBuffers.d_Jw, alignImageBuffers.d_J, alignImageBuffers.d_weights, nPoints);
			
			
			
			
			const double identityScalar = 1.0;
			const double zeroScalar = 0.0;
			
			cublasDgemm_v2(cublasHandle,
			               CUBLAS_OP_T, CUBLAS_OP_N,
			               6, 6, nPoints,
			               &identityScalar,
			               alignImageBuffers.d_Jw, nPoints,
			               alignImageBuffers.d_J, nPoints,
			               &zeroScalar,
			               alignImageBuffers.d_A, 6);
			
			const double negativeIdentityScalar = -1.0;
			
			cublasDgemm_v2(cublasHandle,
			               CUBLAS_OP_T, CUBLAS_OP_N,
			               6, 1, nPoints,
			               &negativeIdentityScalar,
			               alignImageBuffers.d_Jw, nPoints,
			               alignImageBuffers.d_residual, nPoints,
			               &zeroScalar,
			               alignImageBuffers.d_b, 6);
			
			
			Matrix6d A;
			cudaMemcpyAsync(static_cast<void*>(A.data()), static_cast<void*>(alignImageBuffers.d_A), 6*6*sizeof(double), cudaMemcpyDeviceToHost, stream);
			
			Vector6d b;
			cudaMemcpyAsync(static_cast<void*>(b.data()), static_cast<void*>(alignImageBuffers.d_b), 6*sizeof(double), cudaMemcpyDeviceToHost, stream);
			
			CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
			
			
			const Vector6d xi_delta = stepLength * (A + sigma_inv).ldlt().solve(b + sigma_inv*(initialXi - xi));
			
			
			if (std::isnan(xi_delta.sum()))
			{
				std::cerr << "============= XI is NAN =============\n";
				return Vector6d::Zero();
			}
			
			xi = Sophus::SE3d::log(currentPose * Sophus::SE3d::exp(xi_delta));
			
			alignmentStatistics.iterationPerLevel.at(level)++;
			
			const double xi_delta_norm = xi_delta.norm();
			
			if ((1.0/stepLength) * xi_delta_norm <= alignmentSettings.xiConvergenceLength) { break; }
			
			if (stepLength > alignmentSettings.minStepLength) { stepLength *= alignmentSettings.stepLengthReductionFactor; }
		}
		
		CUDA_SAFE_CALL( cudaDestroyTextureObject(gradXTextureObject) );
		CUDA_SAFE_CALL( cudaDestroyTextureObject(gradYTextureObject) );
	}
	
	cublasDestroy_v2(cublasHandle);
	CUDA_SAFE_CALL( cudaStreamDestroy(stream) );
	
	auto end = std::chrono::steady_clock::now();
	alignmentStatistics.elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	
	return xi;
}
