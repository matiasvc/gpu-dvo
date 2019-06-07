//
// Created by matiasvc on 09.04.19.
//

#ifndef GPU_STEREO_DSO_GLUTILS_H
#define GPU_STEREO_DSO_GLUTILS_H

template<typename T, typename U> constexpr size_t offset_of(U T::*member){
	return (char*)&((T*)nullptr->*member) - (char*)nullptr;
}

inline Eigen::Matrix4f homogenous(const Eigen::Vector3f& t, const Eigen::Quaternionf& q){
	Eigen::Matrix4f H;
	H.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
	H.block<3, 1>(0, 3) = t;
	H(3, 0) = 0.0f;
	H(3, 1) = 0.0f;
	H(3, 2) = 0.0f;
	H(3, 3) = 1.0f;
	return H;
}

inline Eigen::Matrix4f projection_matrix(float z_near, float z_far, float x_left, float x_right, float y_top, float y_bottom){
	
	// The resulting "camera" looks in the direction of positive z, with positive x to the right, and positive y down.
	// For a symmetric camera, this implies z_near, z_far, x_right, y_bottom > 0 and x_left, y_top < 0
	
	Eigen::Matrix4f P;
	P.fill(0.0f);
	P(0, 0) = 2.0f * z_near / (x_right - x_left);
	P(0, 2) = -(x_right + x_left)/(x_right - x_left);
	P(1, 1) = -2.0f * z_near / (y_bottom - y_top);
	P(1, 2) = (y_top + y_bottom) / (y_bottom - y_top);
	P(2, 2) = (z_far + z_near) / (z_far - z_near);
	P(2, 3) = -2.0f*z_far*z_near / (z_far - z_near);
	P(3, 2) = 1.0f;
	
	return P;
}

inline Eigen::Matrix4f projection_matrix(float z_near, float z_far, double fx, double fy, double cx, double cy, double width, double height){
	float x_left = -cx/fx * z_near;
	float x_right = (width - cx)/fx * z_near;
	float y_top = -cy/fy * z_near;
	float y_bottom = (height - cy)/fy * z_near;
	return projection_matrix(z_near, z_far, x_left, x_right, y_top, y_bottom);
}

inline Eigen::Matrix4f projection_matrix_symmetric(float z_near, float z_far, int screen_width, int screen_height, float f){
	return projection_matrix(z_near, z_far, f, f, 0.5f*screen_width, 0.5f*screen_height, screen_width, screen_height);
}

#endif //GPU_STEREO_DSO_GLUTILS_H
