cmake_minimum_required(VERSION 3.13)
project(gpu_dvo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-march=native --fast-math -DEIGEN_DONT_VECTORIZE=1 -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT=1 ")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-march=native --fast-math -DEIGEN_DONT_VECTORIZE=1 -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT=1 ")
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-relaxed-constexpr --use_fast_math --gpu-architecture=compute_30 --gpu-code=sm_61 -DEIGEN_DONT_VECTORIZE=1 -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT=1 ")


# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories("${CUDA_INCLUDE_DIRS}")

# VisionWorks
find_package(VisionWorks REQUIRED)

# stb
include_directories(thirdparty/stb)
set(STB_SOURCE_FILES
    thirdparty/stb/stb_image.cpp
    thirdparty/stb/stb_image_write.cpp
    thirdparty/stb/stb_image_resize.cpp)
add_library(stb ${STB_SOURCE_FILES})

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# Boost
find_package(Boost COMPONENTS program_options REQUIRED)

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# glad
include_directories(thirdparty/glad/include)
add_library(glad STATIC thirdparty/glad/src/glad.c)

# glfw
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

# imgui
include_directories(thirdparty/imgui)
set(IMGUI_SOURCE_FILES
    thirdparty/imgui/imgui.cpp
    thirdparty/imgui/imgui_demo.cpp
    thirdparty/imgui/imgui_draw.cpp
    thirdparty/imgui/imgui_impl_glfw.cpp
    thirdparty/imgui/imgui_impl_opengl3.cpp
    thirdparty/imgui/imgui_widgets.cpp
    )
add_library(imgui STATIC ${IMGUI_SOURCE_FILES})

# cuda code
file(GLOB_RECURSE CUDA_SOURCE_FILES src/*.cu)
add_library(cuda_code ${CUDA_SOURCE_FILES})
set_target_properties(cuda_code PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_LIBRARIES
    Eigen3::Eigen
    stb
    cublas
    )

target_link_libraries(cuda_code ${CUDA_LIBRARIES})

# C++ code
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

set(LIBRARIES
    ${VisionWorks_LIBRARIES}
    stdc++fs
    Boost::program_options
    Eigen3::Eigen
    ${Sophus_LIBS}
    ${YAML_CPP_LIBRARIES}
    stb
    glad
    cuda_code
    imgui
    ${GLFW_STATIC_LIBRARIES}
    ${CUDA_LIBRARIES}
)

add_executable(gpu_dvo ${SOURCE_FILES})

# Make shader include file
file(GLOB SHADER_SOURCE_FILES shaders/*.fs shaders/*.vs)
add_custom_target(
		shaders
		COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/generateShaderInclude.py ${SHADER_SOURCE_FILES}
)
add_dependencies(gpu_dvo shaders)
include_directories(shaders)


#target_compile_options(gpu_dvo PRIVATE -Wall -Wextra -pedantic -Werror)
target_link_libraries(gpu_dvo ${LIBRARIES})
