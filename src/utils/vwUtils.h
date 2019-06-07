//
// Created by matiasvc on 20.02.19.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_VWUTILS_H
#define DIRECT_IMAGE_ALIGNMENT_VWUTILS_H

#ifdef NDEBUG // Release

#define NVXIO_ASSERT(cond) ((void)0)
#define NVXCU_SAFE_CALL(nvxcuOp) (nvxcuOp)
#define NVXIO_CUDA_SAFE_CALL(cudaOp) (cudaOp)
#define NVXIO_SAFE_CALL(vxOp) (vxOp)
#define NVXIO_CHECK_REFERENCE(ref) ((void)0)

#else // Debug

#include <NVX/nvxcu.h>

/**
 * \ingroup group_nvxio_utility
 * \brief Throws `std::runtime_error` exception.
 * \param [in] msg A message with content related to the exception.
 * \see nvx_nvxio_api
 */
#define NVXIO_THROW_EXCEPTION(msg) \
	do { \
		std::ostringstream ostr_; \
		ostr_ << msg; \
		throw std::runtime_error(ostr_.str()); \
	} while(0)

/**
 * \ingroup group_nvxio_utility
 * \brief Checks a condition. If the condition is false then it throws `std::runtime_error` exception.
 * \param [in] cond Expression to be evaluated.
 * \see nvx_nvxio_api
 */
#define NVXIO_ASSERT(cond) \
	do \
	{ \
		if (!(cond)) \
		{ \
			NVXIO_THROW_EXCEPTION(#cond << " failure in file " << __FILE__ << " line " << __LINE__); \
		} \
	} while (0)
	

#define NVXCU_SAFE_CALL(nvxcuOp) \
	do \
	{ \
		nvxcu_error_status_e stat = (nvxcuOp); \
		if (stat != NVXCU_SUCCESS) \
		{ \
			NVXIO_THROW_EXCEPTION(#nvxcuOp << " failure [status = " << stat << "]" << " in file " << __FILE__ << " line " << __LINE__); \
		} \
	} while (0)

/**
 * \ingroup group_nvxio_utility
 * \brief Performs a CUDA operation. If the operation fails, then it throws `std::runtime_error` exception.
 * \param [in] cudaOp Specifies a function to be called.
 * The function must have `cudaError_t` return value.
 * \see nvx_nvxio_api
 */
#define NVXIO_CUDA_SAFE_CALL(cudaOp) \
	do \
	{ \
		cudaError_t err = (cudaOp); \
		if (err != cudaSuccess) \
		{ \
			std::ostringstream ostr; \
			ostr << "CUDA Error in " << #cudaOp << __FILE__ << " file " << __LINE__ << " line : " << cudaGetErrorString(err); \
			throw std::runtime_error(ostr.str()); \
		} \
	} while (0)

/**
 * \ingroup group_nvxio_utility
 * \brief Performs an NVX operation. If the operation fails, then it throws `std::runtime_error` exception.
 * \param [in] vxOp A function to be called.
 * The function must have `vx_status` return value.
 * \see nvx_nvxio_api
 */
#define NVXIO_SAFE_CALL(vxOp) \
	do \
	{ \
		vx_status status = (vxOp); \
		if (status != VX_SUCCESS) \
		{ \
			NVXIO_THROW_EXCEPTION(#vxOp << " failure [status = " << status << "]" << " in file " << __FILE__ << " line " << __LINE__); \
		} \
	} while (0)
	

/**
 * \ingroup group_nvxio_utility
 * \brief Checks a reference. If the reference is not valid then it throws `std::runtime_error` exception.
 * \param [in] ref Reference to be checked.
 * \see nvx_nvxio_api
 */
#define NVXIO_CHECK_REFERENCE(ref) \
	NVXIO_ASSERT(ref != 0 && vxGetStatus((vx_reference)ref) == VX_SUCCESS)

#endif // NDEBUG

#endif //DIRECT_IMAGE_ALIGNMENT_VWUTILS_H
