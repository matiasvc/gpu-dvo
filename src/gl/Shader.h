//
// Created by matiasvc on 09.04.19.
//

#ifndef GPU_STEREO_DSO_SHADER_H
#define GPU_STEREO_DSO_SHADER_H


#include "Resource.h"
#include <eigen3/Eigen/Dense>
#include <glad/glad.h>
#include <string>

class Shader {
public:
	Shader(const std::string& vertex_source, const std::string& fragment_source);
	Shader(const std::string& vertex_source, const std::string& geometry_source, const std::string& fragment_source);
	
	void set_uniform(const std::string& name, int value) const;
	void set_uniform(const std::string& name, float value) const;
	void set_uniform(const std::string& name, bool value) const;
	void set_uniform(const std::string& name, const Eigen::Vector2f& value) const;
	void set_uniform(const std::string& name, const Eigen::Vector3f& value) const;
	void set_uniform(const std::string& name, const Eigen::Matrix4f& value) const;
	
	void use() const;

private:
	utils::Resource<unsigned int> compile(const std::string& source_code, const GLenum type);
	utils::Resource<unsigned int> program;
};


#endif //GPU_STEREO_DSO_SHADER_H
