#pragma once

#include <eigen3/Eigen/Core>

struct Line3DVertex {
	Eigen::Vector3f pos;
	Eigen::Vector3f color;
};

struct Line3D {
	Eigen::Vector3d start;
	Eigen::Vector3d end;
	Eigen::Vector3f color;
};
