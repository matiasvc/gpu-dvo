#pragma once

#include <stdint.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <NVX/nvxcu.h>

#include "Resource.h"
#include "Shader.h"
#include "../primitives/Point3D.h"


struct GLPoint3D
{
	Eigen::Vector3f pos;
	Eigen::Vector3f color;
	float size;
};


class PointDrawer3D
{
public:
	PointDrawer3D();
	
	void set(const std::vector<Point3D>& points);
	void set(const nvxcu_plain_array_t& pointArray);
	void draw(const Eigen::Vector3d& cam_pos, const Eigen::Quaterniond& cam_att, const Eigen::Matrix4f& projection);

private:
	size_t m_count;
	utils::Resource<unsigned int> m_vao, m_vbo;
	Shader m_shader;
};



