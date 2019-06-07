#pragma once

#include <algorithm>
#include <NVX/nvxcu.h>

inline bool all_equal_image_size(std::initializer_list<nvxcu_pitch_linear_image_t> images)
{
	return std::all_of(images.begin(), images.end(), [&images](auto& image) -> bool
	{
		return images.begin()->base.width == image.base.width and images.begin()->base.height == image.base.height;
	});
}

inline bool all_equal_image_size(std::initializer_list<nvxcu_pitch_linear_pyramid_t> images)
{
	// Check that all images have equal number of layers
	if (!std::all_of(images.begin(), images.end(), [&images](auto& image) -> bool { return images.begin()->base.num_levels == image.base.num_levels; }))
	{
		return false;
	}
	
	// Check that the size of all layers are equal
	return std::all_of(images.begin(), images.end(), [&images](auto& image) -> bool
	{
		for (int i = 0; i < image.base.num_levels; ++i)
		{
			if (images.begin()->levels[i].base.width != image.levels[i].base.width or images.begin()->levels[i].base.height != image.levels[i].base.height)
			{
				return false;
			}
		}
		
		return true;
	});
}

inline bool all_equal_image_type(std::initializer_list<nvxcu_pitch_linear_image_t> images)
{
	return std::all_of(images.begin(), images.end(), [&images](auto& image) -> bool
	{
		return images.begin()->base.format == image.base.format;
	});
}

template<typename T>
inline bool all_equal(std::initializer_list<T> l)
{
	return std::all_of(l.begin(), l.end(), [&l](auto& e) -> bool { return *l.begin() == e; });
}
