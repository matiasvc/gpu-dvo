//
// Created by matiasvc on 24.02.19.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_NVXCU_DEBUG_H
#define DIRECT_IMAGE_ALIGNMENT_NVXCU_DEBUG_H


#ifdef NDEBUG // Release

#define NVXCU_THROW_EXCEPTION(cond) ((void)0)
#define NVXCU_SAFE_CALL(nvxcuOp) (nvxcuOp)
#define CUDA_SAFE_CALL(cudaOp) (cudaOp)

#else // Debug

#include <sstream>
#include <stdexcept>

#define NVXCU_THROW_EXCEPTION(msg) \
	do { \
		std::ostringstream ostr_; \
		ostr_ << msg; \
		throw std::runtime_error(ostr_.str()); \
	} while(0)

#define NVXCU_SAFE_CALL(nvxcuOp) \
	do \
	{ \
		nvxcu_error_status_e stat = (nvxcuOp); \
		if (stat != NVXCU_SUCCESS) \
		{ \
			NVXCU_THROW_EXCEPTION(#nvxcuOp << " failure [status = " << stat << "]" << " in file " << __FILE__ << " line " << __LINE__); \
		} \
	} while (0)

#define CUDA_SAFE_CALL(cudaOp) \
	do \
	{ \
		cudaError_t err = (cudaOp); \
		if (err != cudaSuccess) \
		{ \
			std::ostringstream ostr; \
			ostr << "CUDA Error in " << #cudaOp << __FILE__ << " file " << __LINE__ << " line : " << "Code: " << err << " = " << cudaGetErrorString(err); \
			throw std::runtime_error(ostr.str()); \
		} \
	} while (0)

#endif // NDEBUG


#endif //DIRECT_IMAGE_ALIGNMENT_NVXCU_DEBUG_H
