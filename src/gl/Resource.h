#ifndef GPU_STEREO_DSO_RESOURCE_H
#define GPU_STEREO_DSO_RESOURCE_H

#include <utility>
#include <functional>
#include <type_traits>

namespace utils {
	
	template<typename T>
	class Resource {
	public:
		
		Resource() = default;
		Resource(T handle, std::function<void(T)> releaser) :
				enabled(true),
				handle(std::move(handle)),
				releaser(std::move(releaser)){
		}
		~Resource(){ if (enabled) releaser(std::move(handle)); }
		
		Resource(const Resource&) = delete;
		Resource& operator=(const Resource&) = delete;
		
		Resource(Resource&& rhs) :
				enabled(rhs.enabled),
				handle(std::move(rhs.handle)),
				releaser(std::move(rhs.releaser)){
			rhs.enabled = false;
		}
		
		Resource& operator=(Resource&& rhs){
			if (enabled) releaser(handle);
			enabled = rhs.enabled;
			handle = std::move(rhs.handle);
			releaser = std::move(rhs.releaser);
			rhs.enabled = false;
			return *this;
		}
		
		operator T() const { return handle; }
		
		bool is_empty() const { return !enabled; }
	
	private:
		bool enabled{ false };
		T handle;
		std::function<void(T)> releaser;
	};
	
	template<typename T, typename L>
	auto make_resource(T handle, L&& releaser){
		return Resource<T>(std::move(handle), std::forward<L>(releaser));
	}
	
	template<typename T, typename L>
	auto make_resource(std::function<void(T&)> creater, L&& releaser){
		T handle;
		creater(handle);
		return Resource<T>(std::move(handle), std::forward<L>(releaser));
	}
	
} // namespace utils

#endif //GPU_STEREO_DSO_RESOURCE_H
