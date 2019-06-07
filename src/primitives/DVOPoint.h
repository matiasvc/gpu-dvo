#pragma once

#include <eigen3/Eigen/Core>

#define NVXCU_TYPE_DVO_POINT 0x700

struct DVOPoint
{
	Eigen::Vector3d pos;
	uint8_t intensity;
};
