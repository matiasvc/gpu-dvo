#pragma once

#include <vector>
#include <utility>
#include <string>
#include <cstdlib>
#include <experimental/filesystem>

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

struct dl_image
{
	~dl_image()
	{
		free(m_data);
	}
	
	void* m_data = nullptr;
	int m_width = 0;
	int m_height = 0;
	int m_channels = 0;
	int m_xPitch = 0;
	int m_yPitch = 0;
};

class DataLoader {
public:
	explicit DataLoader(const std::experimental::filesystem::path datasetPath, const Sophus::SE3d transformation = Sophus::SE3d());
	
	void next();
	bool hasNext() const;
	const dl_image getRGB() const;
	const dl_image getDepth() const;
	const Sophus::SE3d getGroundTruth() const;
	double getTimestamp() const;
	
	int size() const;
	int currentIndex() const;

private:
	const std::experimental::filesystem::path m_dataSetPath;
	const Sophus::SE3d m_transformation;
	
	int m_rgbIndex;
	int m_depthIndex;
	int m_groundtruthIndex;
	
	std::vector<std::pair<double, const std::string>> m_rgbFiles;
	std::vector<std::pair<double, const std::string>> m_depthFiles;
	std::vector<std::pair<double, const Sophus::SE3d>> m_groundTruths;
};
