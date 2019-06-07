#include "PointDrawer3D.h"

#include <glad/glad.h>
#include <cassert>
#include <cuda_gl_interop.h>

#include "glDebug.h"
#include "glUtils.h"
#include "shaders.h"
#include "../primitives/DVOPoint.h"
#include "../utils/nvxcu_debug.h"


PointDrawer3D::PointDrawer3D()
: m_count{0}, m_shader{ point_vs_str, point_fs_str }
{}

// CUDA functions
void copyDVOArrayToGLPointArray(const nvxcu_plain_array_t& pointArray, void* vboPtr, size_t arrayElements, cudaStream_t stream);

void PointDrawer3D::set(const std::vector<Point3D>& points)
{
	if (points.empty()) { return; }
	m_count = points.size();
	
	m_vbo = utils::Resource<unsigned int>(); // destroy vbo before its parent vao
	m_vao = utils::make_resource<uint32_t>([](auto& v){ glGenVertexArrays(1, &v); glCheckError(); }, [](auto v){ glDeleteVertexArrays(1, &v); glCheckError(); });
	m_vbo = utils::make_resource<uint32_t>([](auto& v){ glGenBuffers(1, &v); glCheckError(); }, [](auto v){ glDeleteBuffers(1, &v); glCheckError(); });
	
	std::vector<GLPoint3D> glPoints;
	glPoints.reserve(points.size());
	
	for (auto& point: points) {
		const GLPoint3D glPoint{point.pos.cast<float>(), point.color, point.size};
		glPoints.emplace_back(glPoint);
	}
	
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, glPoints.size()*sizeof(GLPoint3D), &glPoints[0], GL_STATIC_DRAW);
	
	{
		constexpr auto location = 0;
		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(GLPoint3D), (void*)offset_of(&GLPoint3D::pos));
		glEnableVertexAttribArray(location);
	}
	{
		constexpr auto location = 1;
		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(GLPoint3D), (void*)offset_of(&GLPoint3D::color));
		glEnableVertexAttribArray(location);
	}
	{
		constexpr auto location = 2;
		glVertexAttribPointer(location, 1, GL_FLOAT, GL_FALSE, sizeof(GLPoint3D), (void*)offset_of(&GLPoint3D::size));
		glEnableVertexAttribArray(location);
	}
	
	glBindVertexArray(0);
}

void PointDrawer3D::set(const nvxcu_plain_array_t& pointArray)
{
	assert(pointArray.base.item_type == NVXCU_TYPE_DVO_POINT);
	uint32_t h_count = 0;
	CUDA_SAFE_CALL( cudaMemcpy(&h_count, pointArray.num_items_dev_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	m_count = h_count;
	
	if (m_count == 0) { return; }
	
	m_vbo = utils::Resource<unsigned int>(); // destroy vbo before its parent vao
	m_vao = utils::make_resource<uint32_t>([](auto& v){ glGenVertexArrays(1, &v); glCheckError(); }, [](auto v){ glDeleteVertexArrays(1, &v); glCheckError(); });
	m_vbo = utils::make_resource<uint32_t>([](auto& v){ glGenBuffers(1, &v); glCheckError(); }, [](auto v){ glDeleteBuffers(1, &v); glCheckError(); });
	
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, m_count*sizeof(GLPoint3D), nullptr, GL_STATIC_DRAW);
	
	{
		constexpr auto location = 0;
		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(GLPoint3D), (void*)offset_of(&GLPoint3D::pos));
		glEnableVertexAttribArray(location);
	}
	{
		constexpr auto location = 1;
		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(GLPoint3D), (void*)offset_of(&GLPoint3D::color));
		glEnableVertexAttribArray(location);
	}
	{
		constexpr auto location = 2;
		glVertexAttribPointer(location, 1, GL_FLOAT, GL_FALSE, sizeof(GLPoint3D), (void*)offset_of(&GLPoint3D::size));
		glEnableVertexAttribArray(location);
	}
	
	cudaStream_t cudaStream;
	cudaGraphicsResource_t cudaGLResource;
	CUDA_SAFE_CALL( cudaStreamCreate(&cudaStream));
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&cudaGLResource, m_vbo, cudaGraphicsMapFlagsWriteDiscard) );
	
	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &cudaGLResource, cudaStream) );
	void* vboPtr;
	size_t bufferSize;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer(&vboPtr, &bufferSize, cudaGLResource) );
	
	copyDVOArrayToGLPointArray(pointArray, vboPtr, h_count, cudaStream);
	
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &cudaGLResource, cudaStream) );
	
	CUDA_SAFE_CALL( cudaStreamSynchronize(cudaStream) );
	CUDA_SAFE_CALL( cudaStreamDestroy(cudaStream) );
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(cudaGLResource) );
	
	
	glBindVertexArray(0);
}

void PointDrawer3D::draw(const Eigen::Vector3d& cam_pos, const Eigen::Quaterniond& cam_att, const Eigen::Matrix4f& projection)
{
	const Eigen::Vector3f& cam_posf = cam_pos.cast<float>();
	const Eigen::Quaternionf& cam_attf = cam_att.cast<float>();
	
	m_shader.use();
	m_shader.set_uniform("projection", projection);
	Eigen::Matrix4f view_matrix = homogenous(cam_attf.conjugate()*-cam_posf, cam_attf.conjugate());
	m_shader.set_uniform("view", view_matrix);
	
	if (m_count > 0){
		glBindVertexArray(m_vao);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glDrawArrays(GL_POINTS, 0, m_count); glCheckError();
		glBindVertexArray(0);
	}
}
