#include "LineDrawer.h"

#include <glad/glad.h>
#include <stdexcept>

#include "shaders.h"
#include "../primitives/Line3D.h"



LineDrawer::LineDrawer() :
		count{ 0 },
		width{ 1.0f },
		shader{ line_vs_str, line_fs_str }{
}

void LineDrawer::set(const std::vector<Line>& lines, double line_width){
	
	width = static_cast<float>(line_width);
	count = lines.size();
	if (count == 0) return;
	if (width <= 0.0f) throw std::runtime_error("Line width must be larger than zero");
	
	std::vector<Line3DVertex> vertices;
	vertices.reserve(lines.size() * 2);
	for (const auto& line : lines){
		vertices.push_back({ line.start.cast<float>(), line.color });
		vertices.push_back({ line.end.cast<float>(), line.color });
	}
	
	vbo = utils::Resource<unsigned int>(); // destroy vbo before its parent vao
	vao = utils::make_resource<unsigned int>([](auto& x){ glGenVertexArrays(1, &x); }, [](auto x){ glDeleteVertexArrays(1, &x); });
	vbo = utils::make_resource<unsigned int>([](auto& x){ glGenBuffers(1, &x); }, [](auto x){ glDeleteBuffers(1, &x); });
	
	glBindVertexArray(vao);
	
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(vertices[0]), &vertices[0], GL_STATIC_DRAW);
	
	{
		constexpr auto location = 0;
		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (void*)offset_of(&Line3DVertex::pos));
		glEnableVertexAttribArray(location);
	}
	{
		constexpr auto location = 1;
		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (void*)offset_of(&Line3DVertex::color));
		glEnableVertexAttribArray(location);
	}
	
	glBindVertexArray(0);
}

void LineDrawer::draw(const Eigen::Vector3d& _cam_pos, const Eigen::Quaterniond& _cam_att, const Eigen::Matrix4f& projection){
	const Eigen::Vector3f& cam_pos = _cam_pos.cast<float>();
	const Eigen::Quaternionf& cam_att = _cam_att.cast<float>();
	
	shader.use();
	shader.set_uniform("projection", projection);
	Eigen::Matrix4f view_matrix = homogenous(cam_att.conjugate()*-cam_pos, cam_att.conjugate());
	shader.set_uniform("view", view_matrix);
	
	if (count > 0){
		glBindVertexArray(vao);
		glLineWidth(width);
		glDrawArrays(GL_LINES, 0, 2*count);
		glLineWidth(1.5);
		glBindVertexArray(0);
	}
}
