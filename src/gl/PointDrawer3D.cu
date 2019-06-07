#include "PointDrawer3D.h"

#include <cuda_runtime_api.h>
#include <NVX/nvxcu.h>
#include "../primitives/DVOPoint.h"

__global__
void copyDVOArrayToGLPointArrayKernel(DVOPoint* dvoPoints, GLPoint3D* glPoints, size_t arrayElements)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= arrayElements) { return; }
	
	glPoints[index].pos = dvoPoints[index].pos.cast<float>();
	const float intensity = static_cast<float>(dvoPoints[index].intensity)/255.0f;
	glPoints[index].color = Eigen::Vector3f(intensity, intensity, intensity);
	glPoints[index].size = 3.0f;
}

void copyDVOArrayToGLPointArray(const nvxcu_plain_array_t& pointArray, void* vboPtr, size_t arrayElements, cudaStream_t stream)
{
	dim3 blocks(std::min(static_cast<uint32_t>(arrayElements), 1024u), 1, 1);
	dim3 grids(static_cast<uint32_t>(arrayElements) / blocks.x + (static_cast<uint32_t>(arrayElements) % blocks.x == 0u ? 0u : 1u), 1, 1);
	
	copyDVOArrayToGLPointArrayKernel<<<grids, blocks, 0, stream>>>((DVOPoint*)pointArray.dev_ptr, (GLPoint3D*)vboPtr, arrayElements);
}
