#include "DataLoader.h"
#include <iostream>
#include <exception>
#include <stdint.h>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

DataLoader::DataLoader(const std::experimental::filesystem::path datasetPath, const Sophus::SE3d transformation)
		: m_dataSetPath{datasetPath}, m_transformation{transformation}, m_rgbIndex{0}, m_depthIndex{0}, m_groundtruthIndex{0}
{
	const std::experimental::filesystem::path rgbFilePath = datasetPath / "rgb.txt";
	
	std::ifstream rgbFile(rgbFilePath.string());
	if (!rgbFile.is_open()) { throw std::invalid_argument("Unable to open rgb.txt file"); }
	
	
	std::string line;
	const std::string delimiter = " ";
	
	while (getline(rgbFile, line))
	{
		if (line[0] == '#') { continue; }
		const size_t delimiterPos = line.find(delimiter);
		
		const double timestamp = std::stod(line.substr(0, delimiterPos));
		const std::string fileName = line.substr(delimiterPos+1, line.length());
		
		m_rgbFiles.emplace_back(timestamp, fileName);
	}
	
	const std::experimental::filesystem::path depthFilePath = datasetPath / "depth.txt";
	
	std::ifstream depthFile(depthFilePath.string());
	if (!depthFile.is_open()) { throw std::invalid_argument("Unable to open depth.txt file"); }
	
	while (getline(depthFile, line))
	{
		if (line[0] == '#') { continue; }
		const size_t delimiterPos = line.find(delimiter);
		
		const double timestamp = std::stod(line.substr(0, delimiterPos));
		const std::string fileName = line.substr(delimiterPos+1, line.length());
		
		m_depthFiles.emplace_back(timestamp, fileName);
	}
	
	const std::experimental::filesystem::path groundtruthFilePath = datasetPath / "groundtruth.txt";
	
	std::ifstream groudtruthFile(groundtruthFilePath.string());
	if (!groudtruthFile.is_open()) { throw std::invalid_argument("Unable to open groundtruth.txt file"); }
	
	while (getline(groudtruthFile, line))
	{
		if (line[0] == '#') { continue; }
		
		std::stringstream ss(line);
		std::string item;
		
		std::getline(ss, item, ' ');
		double timestamp = std::stod(item);
		
		std::getline(ss, item, ' ');
		double tx = std::stod(item);
		
		std::getline(ss, item, ' ');
		double ty = std::stod(item);
		
		std::getline(ss, item, ' ');
		double tz = std::stod(item);
		
		std::getline(ss, item, ' ');
		double qx = std::stod(item);
		
		std::getline(ss, item, ' ');
		double qy = std::stod(item);
		
		std::getline(ss, item, ' ');
		double qz = std::stod(item);
		
		std::getline(ss, item, ' ');
		double qw = std::stod(item);
		
		const Eigen::Quaterniond orientation(qw, qx, qy, qz);
		const Eigen::Vector3d position(tx, ty, tz);
		const Sophus::SE3d transform(orientation, position);
		
		m_groundTruths.emplace_back(timestamp, transform);
	}
	
	this->next();
}

void DataLoader::next()
{
	m_rgbIndex++;
	
	const double currentTimestamp = m_rgbFiles[m_rgbIndex].first;
	
	while (std::abs(m_depthFiles[m_depthIndex+1].first - currentTimestamp) < std::abs(m_depthFiles[m_depthIndex].first - currentTimestamp) and
	       m_depthIndex < m_depthFiles.size() - 1)
	{
		m_depthIndex++;
	}
	
	while (std::abs(m_groundTruths[m_groundtruthIndex+1].first - currentTimestamp) < std::abs(m_groundTruths[m_groundtruthIndex].first - currentTimestamp) and
	       m_groundtruthIndex < m_groundTruths.size() - 1)
	{
		m_groundtruthIndex++;
	}
	
}

bool DataLoader::hasNext() const
{
	return m_rgbIndex < m_rgbFiles.size();
}

int DataLoader::size() const
{
	return (int)m_rgbFiles.size();
}

int DataLoader::currentIndex() const
{
	return m_rgbIndex;
}

const dl_image DataLoader::getDepth() const
{
	std::experimental::filesystem::path depthImagePath = m_dataSetPath / m_depthFiles[m_depthIndex].second;
	
	dl_image image;
	
	const int depthChannels = 1;
	
	image.m_data = stbi_load_16(depthImagePath.c_str(), &image.m_width, &image.m_height, &image.m_channels, depthChannels);
	image.m_xPitch = depthChannels * sizeof(uint16_t);
	image.m_yPitch = image.m_xPitch * image.m_width;
	
	return image;
}

const dl_image DataLoader::getRGB() const
{
	std::experimental::filesystem::path rgbImagePath = m_dataSetPath / m_rgbFiles[m_rgbIndex].second;
	
	dl_image image;
	
	const int rgbChannels = 3;
	
	image.m_data = stbi_load(rgbImagePath.c_str(), &image.m_width, &image.m_height, &image.m_channels, rgbChannels);
	image.m_xPitch = rgbChannels * sizeof(uint8_t);
	image.m_yPitch = image.m_xPitch * image.m_width;
	
	return image;
}

const Sophus::SE3d DataLoader::getGroundTruth() const
{
	return m_transformation * m_groundTruths[m_groundtruthIndex].second;
}

double DataLoader::getTimestamp() const
{
	return m_rgbFiles[m_rgbIndex].first;
}
