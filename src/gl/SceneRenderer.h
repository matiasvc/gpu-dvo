//
// Created by matiasvc on 08.04.19.
//

#ifndef GPU_STEREO_DSO_SCENERENDERER_H
#define GPU_STEREO_DSO_SCENERENDERER_H

#include <stdint.h>
#include <driver_types.h>
#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>
#include <NVX/nvxcu.h>
#include <vector>

#include "PointDrawer3D.h"
#include "LineDrawer.h"
#include "../primitives/Line3D.h"

#include "../primitives/InstrinsicParameters.h"

class SceneRenderer
{
public:
	explicit SceneRenderer();
	
	void rotate(double pitchDelta, double yawDelta);
	void move(double distanceDelta);
	void moveViewCenter(double forewardsDelta, double rightDelta);
	void drawGrid(int windowWidth, int windowHeight);
	void drawPoints(const nvxcu_pitch_linear_image_t& grayImage, const nvxcu_pitch_linear_image_t& depthImage, const nvxcu_plain_array_t& pointArray, IntrinsicParameters intrinsicParameters,
	                const Sophus::SE3d& transform, int windowWidth, int windowHeight);
	void drawGrountruth(const Sophus::SE3d& transform, int windowWidth, int windowHeight);
	void drawEstimated(const Sophus::SE3d& transform, int windowWidth, int windowHeight);
	
private:
	double m_camDistance;
	Eigen::Vector3d m_camRPY;
	Eigen::Vector3d m_camViewCenter;
	
	
	LineDrawer m_gridDrawer;
	LineDrawer m_axisDrawer;
	LineDrawer m_groundtruthPathDrawer;
	LineDrawer m_groundtruthFrustrumDrawer;
	LineDrawer m_estimatedPathDrawer;
	LineDrawer m_estimatedFrustrumDrawer;
	PointDrawer3D m_pointDrawer;
	
	std::vector<Eigen::Vector3d> m_groundTruthPath;
	std::vector<Eigen::Vector3d> m_estimatedPath;
	
	const Sophus::SE3d getCameraTransform();
	const Eigen::Vector3d getGlobalCamPosition();
	const Eigen::Matrix4f getCamProjection(int windowWidth, int windowHeight);
	const Eigen::Quaterniond getCamOrientation();
};


#endif //GPU_STEREO_DSO_SCENERENDERER_H
