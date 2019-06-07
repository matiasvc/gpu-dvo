#pragma once

#include <eigen3/Eigen/Core>

struct Point3D {
	Eigen::Vector3d pos;
	Eigen::Vector3f color;
	float size;
};
