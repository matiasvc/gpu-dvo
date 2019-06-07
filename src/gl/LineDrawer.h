//
// Created by matiasvc on 09.04.19.
//

#ifndef GPU_STEREO_DSO_LINEDRAWER_H
#define GPU_STEREO_DSO_LINEDRAWER_H

#include "Shader.h"

#include "glUtils.h"
#include <eigen3/Eigen/Dense>

#include <vector>

struct Line {
	Eigen::Vector3d start;
	Eigen::Vector3d end;
	Eigen::Vector3f color;
};

class LineDrawer {
public:
	
	LineDrawer();
	void set(const std::vector<Line>& lines, double line_width);
	void draw(const Eigen::Vector3d& cam_pos, const Eigen::Quaterniond& cam_att, const Eigen::Matrix4f& projection);

private:
	size_t count;
	float width;
	utils::Resource<unsigned int> vao, vbo;
	Shader shader;
};


#endif //GPU_STEREO_DSO_LINEDRAWER_H
