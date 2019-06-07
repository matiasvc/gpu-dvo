//
// Created by matiasvc on 28.03.19.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_LOADMAPFILES_H
#define DIRECT_IMAGE_ALIGNMENT_LOADMAPFILES_H

#include <NVX/nvxcu.h>
#include <experimental/filesystem>
#include <cuda_runtime.h>
#include <fstream>
#include <sys/stat.h>

bool loadMapFiles(const nvxcu_pitch_linear_image_t& map, const std::experimental::filesystem::path& filepath,
                  int width, int height)
{
	if (!std::experimental::filesystem::exists(filepath))
	{
		std::cerr << "Error! Map file does not exist: " << filepath.string() << '\n';
		return false;
	}
	
	const size_t imageSize = width*height*2*sizeof(float);
	
	struct stat results{};
	if (stat(filepath.c_str(), &results) == 0)
	{
		if (results.st_size != imageSize)
		{
			std::cerr << "Error! Unexpected file size of: " << filepath.string() << '\n';
			std::cerr << "Should be: " << imageSize << " but is instead " << results.st_size << '\n';
			return false;
		}
	}
	else { std::cerr << "Error! Unable to read filesize: " << filepath.string() << '\n'; }
	
	char* host_memory = (char*)malloc(imageSize);
	
	std::ifstream mapFile(filepath.string(), std::ios::in | std::ios::binary);
	mapFile.read(host_memory, imageSize);
	mapFile.close();
	
	CUDA_SAFE_CALL( cudaMemcpy2D(
			map.planes[0].dev_ptr, (size_t)map.planes[0].pitch_in_bytes,
			host_memory, (size_t)width*2*sizeof(float),
			(size_t)width*2*sizeof(float), (size_t)height,
			cudaMemcpyHostToDevice)
	);
	
	free(host_memory);
}


#endif //DIRECT_IMAGE_ALIGNMENT_LOADMAPFILES_H
