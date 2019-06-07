//
// Created by matiasvc on 08.04.19.
//

#include "SceneRenderer.h"

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cmath>

#include "glUtils.h"
#include "glDebug.h"
#include "../utils/nvxcu_utils.h"
#include "../primitives/InstrinsicParameters.h"
#include "../cuda/imageUtils.h"

SceneRenderer::SceneRenderer()
: m_camDistance{15.0}, m_camRPY{Eigen::Vector3d(-M_PI_4, M_PI_4, 0.0)}, m_camViewCenter{Eigen::Vector3d::Zero()}
{
	// Grid
	std::vector<Line> gridLines;
	gridLines.reserve(20);
	for (int x = -20; x <= 20; ++x)
	{
		Line line{Eigen::Vector3d(double(x), 0.0, -20.0),
		          Eigen::Vector3d(double(x), 0.0,  20.0),
		          Eigen::Vector3f(0.5, 0.5, 0.5)};
		gridLines.emplace_back(std::move(line));
	}
	
	for (int z = -20; z <= 20; ++z)
	{
		Line line{Eigen::Vector3d(-20.0, 0.0, double(z)),
		          Eigen::Vector3d( 20.0, 0.0, double(z)),
		          Eigen::Vector3f(  0.5, 0.5, 0.5)};
		gridLines.emplace_back(std::move(line));
	}
	
	m_gridDrawer.set(gridLines, 1.0);
	
	// Axis
	std::vector<Line> axisLines{
			Line{Eigen::Vector3d(0.0, 0.0, 0.0),
			     Eigen::Vector3d(1.0, 0.0, 0.0),
			     Eigen::Vector3f(1.0, 0.0, 0.0)},
			
			Line{Eigen::Vector3d(0.0, 0.0, 0.0),
			     Eigen::Vector3d(0.0, 1.0, 0.0),
			     Eigen::Vector3f(0.0, 1.0, 0.0)},
			
			Line{Eigen::Vector3d(0.0, 0.0, 0.0),
			     Eigen::Vector3d(0.0, 0.0, 1.0),
			     Eigen::Vector3f(0.0, 0.0, 1.0)}
	};
	
	m_axisDrawer.set(axisLines, 2.0);
}

void SceneRenderer::rotate(double pitchDelta, double yawDelta)
{
	m_camRPY += Eigen::Vector3d(pitchDelta, yawDelta, 0.0);
}

void SceneRenderer::move(double distanceDelta)
{
	m_camDistance += distanceDelta;
	
	if (m_camDistance < 0.1f) { m_camDistance = 0.1f; }
}

void SceneRenderer::moveViewCenter(double forewardsDelta, double rightDelta)
{
	const double yaw = m_camRPY.x();
	
	m_camViewCenter += Eigen::Vector3d(rightDelta*std::sin(yaw) - forewardsDelta*std::cos(yaw),
	                                   0,
	                                   rightDelta*std::cos(yaw) + forewardsDelta*std::sin(yaw));
}

const Sophus::SE3d SceneRenderer::getCameraTransform()
{
	const Sophus::SE3d transform(getCamOrientation(), getGlobalCamPosition());
	return transform;
}

const Eigen::Vector3d SceneRenderer::getGlobalCamPosition()
{
	const Eigen::Vector3d camPosition(0.0, 0.0, -m_camDistance);
	const Eigen::Quaterniond camOrientation = this->getCamOrientation();
	
	return camOrientation.toRotationMatrix() * camPosition + m_camViewCenter;
}

const Eigen::Matrix4f SceneRenderer::getCamProjection(int windowWidth, int windowHeight)
{
	return projection_matrix_symmetric(0.001f, 100.0f, windowWidth, windowHeight, 0.25f*(windowWidth + windowHeight));
}

const Eigen::Quaterniond SceneRenderer::getCamOrientation()
{
	return Eigen::AngleAxisd( m_camRPY.x(), Eigen::Vector3d::UnitY()) *
	       Eigen::AngleAxisd(-m_camRPY.y(), Eigen::Vector3d::UnitX());
}

void SceneRenderer::drawGrid(int windowWidth, int windowHeight)
{
	const Eigen::Matrix4f camProjection = this->getCamProjection(windowWidth, windowHeight);
	const Eigen::Quaterniond camOrientation = this->getCamOrientation();
	const Eigen::Vector3d globalCamPosition = this->getGlobalCamPosition();
	
	m_gridDrawer.draw(globalCamPosition, camOrientation, camProjection);
	m_axisDrawer.draw(globalCamPosition, camOrientation, camProjection);
}

void SceneRenderer::drawPoints(const nvxcu_pitch_linear_image_t& grayImage, const nvxcu_pitch_linear_image_t& depthImage, const nvxcu_plain_array_t& pointArray,
                               IntrinsicParameters intrinsicParameters, const Sophus::SE3d& transform, int windowWidth, int windowHeight)
{
	const Eigen::Matrix4f camProjection = this->getCamProjection(windowWidth, windowHeight);
	const Sophus::SE3d pointTransform = transform.inverse() * this->getCameraTransform();
	
	projectDepthPixels(grayImage, depthImage, pointArray, intrinsicParameters, nullptr);
	m_pointDrawer.set(pointArray);
	m_pointDrawer.draw(pointTransform.translation(), pointTransform.unit_quaternion(), camProjection);
}

void SceneRenderer::drawGrountruth(const Sophus::SE3d& transform, int windowWidth, int windowHeight)
{
	const Eigen::Matrix4f camProjection = this->getCamProjection(windowWidth, windowHeight);
	const Eigen::Quaterniond camOrientation = this->getCamOrientation();
	const Eigen::Vector3d globalCamPosition = this->getGlobalCamPosition();
	
	m_groundTruthPath.emplace_back(transform.translation());
	if (m_groundTruthPath.size() > 1)
	{
		std::vector<Line> groundtruthLines;
		for (int i = 1; i < m_groundTruthPath.size(); ++i)
		{
			Line line{m_groundTruthPath[i-1], m_groundTruthPath[i], Eigen::Vector3f(1.0f, 0.1f, 0.1f)};
			groundtruthLines.emplace_back(std::move(line));
		}
		
		m_groundtruthPathDrawer.set(groundtruthLines, 1.0f);
		m_groundtruthPathDrawer.draw(globalCamPosition, camOrientation, camProjection);
	}
	
	const Eigen::Matrix3d R = transform.rotationMatrix();
	const Eigen::Vector3d& t = transform.translation();
	
	const Eigen::Vector3d center     = R * Eigen::Vector3d( 0,    0, 0) + t;
	const Eigen::Vector3d upperLeft  = R * Eigen::Vector3d(-1, -0.7, 1) + t;
	const Eigen::Vector3d upperRight = R * Eigen::Vector3d( 1, -0.7, 1) + t;
	const Eigen::Vector3d lowerLeft  = R * Eigen::Vector3d(-1,  0.7, 1) + t;
	const Eigen::Vector3d LowerRight = R * Eigen::Vector3d( 1,  0.7, 1) + t;
	const Eigen::Vector3d XAxis      = R * Eigen::Vector3d(0.25, 0.0, 0.0) + t;
	const Eigen::Vector3d YAxis      = R * Eigen::Vector3d(0.0, 0.25, 0.0) + t;
	const Eigen::Vector3d ZAxis      = R * Eigen::Vector3d(0.0, 0.0, 0.25) + t;
	
	const Eigen::Vector3f frustrumColor(1.0f, 0.1f, 0.1f);
	
	std::vector<Line> frustrumLines{
		Line{center, upperLeft, frustrumColor},
		Line{center, upperRight, frustrumColor},
		Line{center, lowerLeft, frustrumColor},
		Line{center, LowerRight, frustrumColor},
		
		Line{upperLeft, upperRight, frustrumColor},
		Line{upperRight, LowerRight, frustrumColor},
		Line{LowerRight, lowerLeft, frustrumColor},
		Line{lowerLeft, upperLeft, frustrumColor},
		Line{t, XAxis, Eigen::Vector3f(1.0, 0.0, 0.0)},
		Line{t, YAxis, Eigen::Vector3f(0.0, 1.0, 0.0)},
		Line{t, ZAxis, Eigen::Vector3f(0.0, 0.0, 1.0)}
	};
	
	m_groundtruthFrustrumDrawer.set(frustrumLines, 1.5f);
	m_groundtruthFrustrumDrawer.draw(globalCamPosition, camOrientation, camProjection);
}

void SceneRenderer::drawEstimated(const Sophus::SE3d& transform, int windowWidth, int windowHeight)
{
	const Eigen::Matrix4f camProjection = this->getCamProjection(windowWidth, windowHeight);
	const Eigen::Quaterniond camOrientation = this->getCamOrientation();
	const Eigen::Vector3d globalCamPosition = this->getGlobalCamPosition();
	
	m_estimatedPath.emplace_back(transform.translation());
	if (m_estimatedPath.size() > 1)
	{
		std::vector<Line> groundtruthLines;
		for (int i = 1; i < m_estimatedPath.size(); ++i)
		{
			Line line{m_estimatedPath[i-1], m_estimatedPath[i], Eigen::Vector3f(0.1f, 0.1f, 1.0f)};
			groundtruthLines.emplace_back(std::move(line));
		}
		
		m_estimatedPathDrawer.set(groundtruthLines, 1.0f);
		m_estimatedPathDrawer.draw(globalCamPosition, camOrientation, camProjection);
	}
	
	const Eigen::Matrix3d R = transform.rotationMatrix();
	const Eigen::Vector3d& t = transform.translation();
	
	const Eigen::Vector3d center     = R * Eigen::Vector3d( 0,    0, 0) + t;
	const Eigen::Vector3d upperLeft  = R * Eigen::Vector3d(-1, -0.7, 1) + t;
	const Eigen::Vector3d upperRight = R * Eigen::Vector3d( 1, -0.7, 1) + t;
	const Eigen::Vector3d lowerLeft  = R * Eigen::Vector3d(-1,  0.7, 1) + t;
	const Eigen::Vector3d LowerRight = R * Eigen::Vector3d( 1,  0.7, 1) + t;
	const Eigen::Vector3d XAxis      = R * Eigen::Vector3d(0.25, 0.0, 0.0) + t;
	const Eigen::Vector3d YAxis      = R * Eigen::Vector3d(0.0, 0.25, 0.0) + t;
	const Eigen::Vector3d ZAxis      = R * Eigen::Vector3d(0.0, 0.0, 0.25) + t;
	
	const Eigen::Vector3f frustrumColor(0.1f, 0.1f, 1.0f);
	
	std::vector<Line> frustrumLines{
			Line{center, upperLeft, frustrumColor},
			Line{center, upperRight, frustrumColor},
			Line{center, lowerLeft, frustrumColor},
			Line{center, LowerRight, frustrumColor},
			
			Line{upperLeft, upperRight, frustrumColor},
			Line{upperRight, LowerRight, frustrumColor},
			Line{LowerRight, lowerLeft, frustrumColor},
			Line{lowerLeft, upperLeft, frustrumColor},
			Line{t, XAxis, Eigen::Vector3f(1.0, 0.0, 0.0)},
			Line{t, YAxis, Eigen::Vector3f(0.0, 1.0, 0.0)},
			Line{t, ZAxis, Eigen::Vector3f(0.0, 0.0, 1.0)}
	};
	
	m_estimatedFrustrumDrawer.set(frustrumLines, 1.5f);
	m_estimatedFrustrumDrawer.draw(globalCamPosition, camOrientation, camProjection);
}

