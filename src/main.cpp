#include <iostream>
#include <cassert>
#include <stdexcept>
#include <numeric>
#include <iomanip>
#include <experimental/filesystem>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <NVX/nvxcu.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <boost/program_options.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "imageAlignment.h"
#include "gl/glDebug.h"
#include "utils/DataLoader.h"
#include "utils/nvxcu_utils.h"
#include "utils/nvxcu_debug.h"
#include "utils/nvxcu_imageRenderer.h"
#include "utils/nvxcu_pyramidRenderer.h"
#include "utils/loadMapFiles.h"
#include "utils/nvxcu_imageSave.h"
#include "gl/SceneRenderer.h"
#include "primitives/DVOPoint.h"
#include "primitives/InstrinsicParameters.h"


#include "cuda/cudaUtils.h"
#include "cuda/imageUtils.h"

// Ignore int to void* cast used by ImGui::Image
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void initImGUI(GLFWwindow* window)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	
	const char* glsl_version = "#version 330 core";
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);
}

GLFWwindow* initGLFW()
{
	glfwSetErrorCallback(glfw_error_callback);
	
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	GLFWwindow* window = glfwCreateWindow(1280, 720, "Direct Image Alignment", nullptr, nullptr);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync
	
	gladLoadGL();
	
	return window;
}

void populateGrayImage(const nvxcu_pitch_linear_pyramid_t& grayPyramid, const nvxcu_pitch_linear_pyramid_t& gradXPyramid, const nvxcu_pitch_linear_pyramid_t& gradYPyramid,
		const nvxcu_pitch_linear_image_t& tmpRGBImage, const dl_image& hostImage, const nvxcu_stream_exec_target_t& exec_target, const nvxcu_border_t& border)
{
	CUDA_SAFE_CALL( cudaMemcpy2D(
			tmpRGBImage.planes[0].dev_ptr, (size_t)tmpRGBImage.planes[0].pitch_in_bytes,
			hostImage.m_data, (size_t)hostImage.m_yPitch,
			(size_t)hostImage.m_yPitch,
			(size_t)hostImage.m_height,
			cudaMemcpyHostToDevice)
	);
	
	NVXCU_SAFE_CALL( nvxcuColorConvert(&tmpRGBImage.base, &grayPyramid.levels[0].base, NVXCU_COLOR_SPACE_NONE, NVXCU_CHANNEL_RANGE_FULL, &exec_target.base) );
	NVXCU_SAFE_CALL( nvxcuGaussianPyramid(&grayPyramid.base, nullptr, &border, &exec_target.base) );
	
	calculateGradientPyramids(grayPyramid, gradXPyramid, gradYPyramid, border, exec_target);
}

void populateDepthImage(const nvxcu_pitch_linear_pyramid_t& depthPyramid, const dl_image& hostImage, const nvxcu_stream_exec_target_t& exec_target)
{
	CUDA_SAFE_CALL( cudaMemcpy2D(
			depthPyramid.levels[0].planes[0].dev_ptr, (size_t)depthPyramid.levels[0].planes[0].pitch_in_bytes,
			hostImage.m_data, (size_t)hostImage.m_yPitch,
			(size_t)hostImage.m_yPitch,
			(size_t)hostImage.m_height,
			cudaMemcpyHostToDevice)
	);
	
	calculateDepthPyramid(depthPyramid, exec_target.stream);
}

struct Options
{
	std::string datasetPath;
	std::string trajectoryFilePath;
	std::string groundtruthFilePath;
	std::string statisticsFilePath;
	std::array<double, 4> cameraParameters;
	bool disableGui;
	bool verbose;
	uint32_t maxIterations;
	double motionPrior;
	double xiConvergenceLength;
	double initialStepLength;
	double stepLengthReductionFactor;
	double minStepLength;
	uint32_t skipFrames;
};

bool parseArgs(int argc, char *argv[], Options& opts)
{
	namespace po = boost::program_options;
	
	po::options_description description("Allowed options");
	description.add_options()
			("help,h", "Show help")
			("no-gui", "Disable GUI")
			("verbose,v", "Verbose")
			("dataset,i", po::value<std::string>(&opts.datasetPath)->required(), "Path of dataset directory")
			("trajectory-path", po::value<std::string>(&opts.trajectoryFilePath), "Path of trajectory output file")
			("groundtruth-path", po::value<std::string>(&opts.groundtruthFilePath), "Path of trajectory output file")
			("statistics-path", po::value<std::string>(&opts.statisticsFilePath), "Path of trajectory output file")
			("camera-parameters,p", po::value<std::vector<double>>()->multitoken()->required(), "Camera parameters: fx fy cx cy")
			("max-iterations", po::value<uint32_t>(&opts.maxIterations)->default_value(20), "Maximum number of iterations per level")
			("motion-prior-stddev", po::value<double>(&opts.motionPrior)->default_value(0.05),"Standard deviation of the motion prior")
			("convergence-length", po::value<double>(&opts.xiConvergenceLength)->default_value(2.5e-4), "Minimum length of delta xi, before the optimization stops")
			("initial-step-length", po::value<double>(&opts.initialStepLength)->default_value(0.8), "Initial step length multiplier")
			("step-length-reduction-factor", po::value<double>(&opts.stepLengthReductionFactor)->default_value(1 - 8e-2), "Factor to shorten step length each iteration")
			("minimum-step-length", po::value<double>(&opts.minStepLength)->default_value(0.3), "Minimum step length where reduction stops")
			("skip-frames", po::value<uint32_t>(&opts.skipFrames)->default_value(0), "Skip the given number of frames before starting");
	
	po::variables_map vm;
	
	try
	{
		po::store(po::command_line_parser(argc, argv).options(description).style(po::command_line_style::unix_style).run(), vm);
		
		if (vm.count("help"))
		{
			std::cout << description << '\n';
			return false;
		}
		
		const std::vector<double> cameraParameters = vm["camera-parameters"].as<std::vector<double>>();
		
		if (!vm["camera-parameters"].empty() && cameraParameters.size() == 4)
		{
			opts.cameraParameters[0] = cameraParameters[0];
			opts.cameraParameters[1] = cameraParameters[1];
			opts.cameraParameters[2] = cameraParameters[2];
			opts.cameraParameters[3] = cameraParameters[3];
		}
		else
		{
			std::cerr << "ERROR: Invalid number of camera parameters. Expected 4, got " << cameraParameters.size() << '\n';
			std::cout << description << '\n';
			return false;
		}
		
		opts.disableGui = vm.count("no-gui") != 0;
		opts.verbose = vm.count("verbose") != 0;
		
		po::notify(vm);
	}
	catch (std::exception &e)
	{
		std::cerr << "ERROR: " << e.what() << '\n';
		std::cout << description << '\n';
		return false;
	}
	
	return true;
}

int main(int argc, char* argv[])
{
	Options opts{};
	if (!parseArgs(argc, argv, opts))
	{
		return 1;
	}
	
	GLFWwindow* window;
	if (!opts.disableGui)
	{
		window = initGLFW();
		initImGUI(window);
	}
	
	// Dataloader Setup
	const std::experimental::filesystem::path datasetPath(opts.datasetPath);
	const Eigen::Quaterniond datasetRotation = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()) *
	                                           Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
	const Eigen::Vector3d datasetTranslation = Eigen::Vector3d::Zero();
	
	const Sophus::SE3d datasetTransform(datasetRotation, datasetTranslation);
	const Sophus::SE3d inverseDatasetTransform = datasetTransform.inverse();
	DataLoader datasetLoader(datasetPath, datasetTransform);
	
	for (int i = 0; i < opts.skipFrames; ++i)
	{
		datasetLoader.next();
	}
	
	const IntrinsicParameters intrinsicParameters{opts.cameraParameters[0], opts.cameraParameters[1], opts.cameraParameters[2], opts.cameraParameters[3]};
	
	//const IntrinsicParameters intrinsicParameters{517.3, 516.5, 318.6, 255.3}; // Freiburg 1
	//const IntrinsicParameters intrinsicParameters{520.9, 521.0, 325.1, 249.7}; // Freiburg 2
	//const IntrinsicParameters intrinsicParameters{535.4, 539.2, 320.1, 247.6}; // Freiburg 3
	
	const AlignmentSettings alignmentSettings = {
			.maxIterations = opts.maxIterations,
			.motionPrior = opts.motionPrior,
			.xiConvergenceLength = opts.xiConvergenceLength,
			.initialStepLength = opts.initialStepLength,
			.stepLengthReductionFactor = opts.stepLengthReductionFactor,
			.minStepLength = opts.minStepLength
	};
	
	nvxcu_stream_exec_target_t exec_target;
	exec_target.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
	cudaStreamCreate(&exec_target.stream);
	cudaGetDeviceProperties(&exec_target.dev_prop, 0);
	
	nvxcu_border_t replicateBorder;
	replicateBorder.mode = NVXCU_BORDER_MODE_REPLICATE;
	
	
	const uint32_t imageWidth = 640;
	const uint32_t imageHeight = 480;
	const size_t levels = 5;
	
	
	uint32_t totalFrames = 0;
	std::array<uint32_t, levels> totalIterationsPerLevel{};
	std::array<uint32_t, levels> maxIterationsPerLevel{};
	long totalElapsedMicroseconds = 0;
	long maxElapsedMicroseconds = 0;
	
	nvxcu_pitch_linear_image_container d_tmpRBGImage(createImage(imageWidth, imageHeight, NVXCU_DF_IMAGE_RGB));
	
	nvxcu_linear_pyramid_container d_currentGrayPyramid(createPyramidHalfScale(imageWidth, imageHeight, NVXCU_DF_IMAGE_U8, levels));
	nvxcu_linear_pyramid_container d_currentGradXPyramid(createPyramidHalfScale(imageWidth, imageHeight, NVXCU_DF_IMAGE_S16, levels));
	nvxcu_linear_pyramid_container d_currentGradYPyramid(createPyramidHalfScale(imageWidth, imageHeight, NVXCU_DF_IMAGE_S16, levels));
	
	nvxcu_linear_pyramid_container d_currentDepthPyramid(createPyramidHalfScale(imageWidth, imageHeight, NVXCU_DF_IMAGE_U16, levels));
	
	nvxcu_linear_pyramid_container d_previousGrayPyramid(createPyramidHalfScale(imageWidth, imageHeight, NVXCU_DF_IMAGE_U8, levels));
	nvxcu_linear_pyramid_container d_previousDepthPyramid(createPyramidHalfScale(imageWidth, imageHeight, NVXCU_DF_IMAGE_U16, levels));
	
	nvxcu_plain_array_container d_pointArray(createPlainArray<DVOPoint>(imageWidth*imageHeight, (nvxcu_array_item_type_e)NVXCU_TYPE_DVO_POINT));
	
	std::unique_ptr<nvxcuPyramidRenderer> grayPyramidRendererPtr;
	std::unique_ptr<nvxcuPyramidRenderer> gradXPyramidRendererPtr;
	std::unique_ptr<nvxcuPyramidRenderer> gradYPyramidRendererPtr;
	
	std::unique_ptr<nvxcuPyramidRenderer> depthPyramidRendererPtr;
	
	std::unique_ptr<SceneRenderer> sceneRendererPtr;
	
	if (!opts.disableGui)
	{
		grayPyramidRendererPtr = std::make_unique<nvxcuPyramidRenderer>(imageWidth, imageHeight, NVXCU_DF_IMAGE_U8, levels);
		gradXPyramidRendererPtr = std::make_unique<nvxcuPyramidRenderer>(imageWidth, imageHeight, NVXCU_DF_IMAGE_S16, levels);
		gradYPyramidRendererPtr = std::make_unique<nvxcuPyramidRenderer>(imageWidth, imageHeight, NVXCU_DF_IMAGE_S16, levels);
		
		depthPyramidRendererPtr = std::make_unique<nvxcuPyramidRenderer>(imageWidth, imageHeight, NVXCU_DF_IMAGE_U16, levels);
		
		sceneRendererPtr = std::make_unique<SceneRenderer>();
	}
	
	nvxcu_tmp_buf_size_t alignImageTmpSize = alignImagesBuffSize(imageWidth, imageHeight);
	nvxcu_tmp_buf_size_t gaussian_tmp_size = nvxcuGaussianPyramid_GetBufSize(imageWidth, imageHeight, levels, NVXCU_SCALE_PYRAMID_HALF, &replicateBorder, &exec_target.dev_prop);
	nvxcu_tmp_buf_container tmp_buf(createTmpBuf({gaussian_tmp_size, alignImageTmpSize}));
	
	AlignImageBuffers alignImageBuffers = createAlignImageBuffers(tmp_buf.m_contained.dev_ptr, imageWidth, imageHeight);
	
	
	// Populate first frame into the previous images
	{
		const auto h_rgbImage = datasetLoader.getRGB();
		const auto h_depthImage = datasetLoader.getDepth();
		
		CUDA_SAFE_CALL( cudaMemcpy2D(
				d_tmpRBGImage.m_contained.planes[0].dev_ptr, (size_t)d_tmpRBGImage.m_contained.planes[0].pitch_in_bytes,
				h_rgbImage.m_data, (size_t)h_rgbImage.m_yPitch,
				(size_t)h_rgbImage.m_yPitch,
				(size_t)h_rgbImage.m_height,
				cudaMemcpyHostToDevice)
		);
		
		NVXCU_SAFE_CALL( nvxcuColorConvert(&d_tmpRBGImage.m_contained.base, &d_previousGrayPyramid.m_contained.levels[0].base, NVXCU_COLOR_SPACE_NONE, NVXCU_CHANNEL_RANGE_FULL, &exec_target.base) );
		NVXCU_SAFE_CALL( nvxcuGaussianPyramid(&d_previousGrayPyramid.m_contained.base, nullptr, &replicateBorder, &exec_target.base) );
		
		populateDepthImage(d_previousDepthPyramid.m_contained, h_depthImage, exec_target);
	}
	Sophus::SE3d estimatedPose = datasetLoader.getGroundTruth();
	Sophus::SE3d previousGroundTruthPose = estimatedPose;
	
	std::vector<std::array<double, 8>> estimatedPosePath;
	std::vector<std::array<double, 8>> groundtruthPosePath;
	
	Eigen::Matrix<double, 6, 1> previousXi = Eigen::Matrix<double, 6, 1>::Zero();
	
	datasetLoader.next();
	
	int frame = 1;
	
	
	while(datasetLoader.hasNext())
	{
		if (!opts.disableGui)
		{
			glfwPollEvents();
			// ImGui Render
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			
			// OpenGL Render
			glfwMakeContextCurrent(window);
		}
		
		
		const auto h_rgbImage = datasetLoader.getRGB();
		const auto h_depthImage = datasetLoader.getDepth();
		
		populateGrayImage(d_currentGrayPyramid.m_contained, d_currentGradXPyramid.m_contained, d_currentGradYPyramid.m_contained, d_tmpRBGImage.m_contained, h_rgbImage, exec_target, replicateBorder);
		populateDepthImage(d_currentDepthPyramid.m_contained, h_depthImage, exec_target);
		
		cudaStreamSynchronize(exec_target.stream);
		AlignmentStatistics<levels> alignmentStatistics{};
		
		Eigen::Matrix<double, 6, 1> xi = alignImages(d_currentGrayPyramid.m_contained, d_currentGradXPyramid.m_contained, d_currentGradYPyramid.m_contained, d_previousGrayPyramid.m_contained, d_previousDepthPyramid.m_contained, d_pointArray.m_contained,
				intrinsicParameters, previousXi, alignImageBuffers, alignmentSettings, alignmentStatistics);
		
		const Sophus::SE3d groundTruthPose = datasetLoader.getGroundTruth();
		const Sophus::SE3d groundTruthDeltaPose =  groundTruthPose * previousGroundTruthPose.inverse();
		
		const Sophus::SE3d estimatedDeltaPose = Sophus::SE3d::exp(xi).inverse();
		estimatedPose = estimatedPose*estimatedDeltaPose;
		
		
		if (opts.verbose)
		{
			std::cout << ">>>>>>>>>>>> Frame: " << frame << " Iterations: "
			          << alignmentStatistics.iterationPerLevel[0] << ", "
			          << alignmentStatistics.iterationPerLevel[1] << ", "
			          << alignmentStatistics.iterationPerLevel[2] << ", "
			          << alignmentStatistics.iterationPerLevel[3] << ", "
			          << alignmentStatistics.iterationPerLevel[4] << " Total: "
			          << std::accumulate(alignmentStatistics.iterationPerLevel.begin(), alignmentStatistics.iterationPerLevel.end(), 0u) << '\n';
		}
		
		
		if (!opts.trajectoryFilePath.empty())
		{
			const Eigen::Vector3d& poseTransform = estimatedPose.translation();
			const Eigen::Quaterniond& poseQuaternion = estimatedPose.unit_quaternion();
			estimatedPosePath.emplace_back(std::array<double, 8>{datasetLoader.getTimestamp(), poseTransform.x(), poseTransform.y(), poseTransform.z(), poseQuaternion.x(), poseQuaternion.y(), poseQuaternion.z(), poseQuaternion.w()});
		}
		
		if(!opts.groundtruthFilePath.empty())
		{
			const Eigen::Vector3d& groundtruthPoseTransform = groundTruthPose.translation();
			const Eigen::Quaterniond& groundtruthPoseQuaternion = groundTruthPose.unit_quaternion();
			groundtruthPosePath.emplace_back(std::array<double, 8>{datasetLoader.getTimestamp(), groundtruthPoseTransform.x(), groundtruthPoseTransform.y(), groundtruthPoseTransform.z(), groundtruthPoseQuaternion.x(), groundtruthPoseQuaternion.y(), groundtruthPoseQuaternion.z(), groundtruthPoseQuaternion.w()});
		}
		
		
		for (int level = 0; level < levels; ++level)
		{
			totalIterationsPerLevel[level] += alignmentStatistics.iterationPerLevel[level];
			maxIterationsPerLevel[level] = std::max(maxIterationsPerLevel[level], alignmentStatistics.iterationPerLevel[level]);
		}
		
		totalElapsedMicroseconds += alignmentStatistics.elapsedMicroseconds;
		maxElapsedMicroseconds = std::max(maxElapsedMicroseconds, alignmentStatistics.elapsedMicroseconds);
		
		totalFrames++;
		frame++;
		
		if (!opts.disableGui)
		{
			int display_w, display_h;
			glfwGetFramebufferSize(window, &display_w, &display_h);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			
			glViewport(0, 0, display_w, display_h);
			
			glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			
			
			sceneRendererPtr->drawPoints(d_currentGrayPyramid.m_contained.levels[0], d_currentDepthPyramid.m_contained.levels[0], d_pointArray.m_contained, intrinsicParameters, estimatedPose, display_w, display_h);
			sceneRendererPtr->drawGrid(display_w, display_h);
			sceneRendererPtr->drawGrountruth(groundTruthPose, display_w, display_h);
			sceneRendererPtr->drawEstimated(estimatedPose, display_w, display_h);
			
			
			int width, height;
			glfwGetWindowSize(window, &width, &height);
			ImGuiIO& io = ImGui::GetIO();
			
			// Right mouse-button drag
			if (ImGui::IsMouseDragging(0) && !ImGui::IsAnyWindowHovered())
			{
				ImVec2 vec = io.MouseDelta;
				const float rotateSpeed = 0.003f;
				sceneRendererPtr->rotate(vec.x*rotateSpeed, vec.y*rotateSpeed);
			}
			
			// Left mouse-button drag
			if (ImGui::IsMouseDragging(1) && !ImGui::IsAnyWindowHovered())
			{
				ImVec2 vec = io.MouseDelta;
				const float moveSpeed = 0.008f;
				sceneRendererPtr->moveViewCenter(vec.x*moveSpeed, vec.y*moveSpeed);
			}
			
			// Mouse wheel
			if (!ImGui::IsAnyWindowHovered())
			{
				const float scroll = io.MouseWheel;
				const float zoomSpeed = 0.5f;
				
				if (scroll != 0.0f)
				{
					sceneRendererPtr->move(scroll*zoomSpeed);
				}
			}
			
			
			if(ImGui::Begin("Depth"))
			{
				depthPyramidRendererPtr->drawPyramid(d_currentDepthPyramid.m_contained);
				for (unsigned int level = 0; level < levels; ++level)
				{
					ImGui::Image((void*)depthPyramidRendererPtr->getRenderTexture(level), ImVec2(imageWidth/2.0f, imageHeight/2.0f));
				}
				
			} ImGui::End();
			
			if (ImGui::Begin("Image"))
			{
				grayPyramidRendererPtr->drawPyramid(d_currentGrayPyramid.m_contained);
				gradXPyramidRendererPtr->drawPyramid(d_currentGradXPyramid.m_contained);
				gradYPyramidRendererPtr->drawPyramid(d_currentGradYPyramid.m_contained);
				for (unsigned int level = 0; level < levels; ++level)
				{
					ImGui::Image((void*)grayPyramidRendererPtr->getRenderTexture(level), ImVec2(imageWidth/2.0f, imageHeight/2.0f));
					ImGui::SameLine();
					ImGui::Image((void*)gradXPyramidRendererPtr->getRenderTexture(level), ImVec2(imageWidth/2.0f, imageHeight/2.0f));
					ImGui::SameLine();
					ImGui::Image((void*)gradYPyramidRendererPtr->getRenderTexture(level), ImVec2(imageWidth/2.0f, imageHeight/2.0f));
				}
				
			} ImGui::End();
			
			
			// Rendering
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			
			glfwMakeContextCurrent(window);
			glfwSwapBuffers(window);
		}
		
		
		std::swap(d_currentGrayPyramid.m_contained, d_previousGrayPyramid.m_contained);
		std::swap(d_currentDepthPyramid.m_contained, d_previousDepthPyramid.m_contained);
		datasetLoader.next();
		previousGroundTruthPose = groundTruthPose;
		previousXi = xi;
		
		if(!opts.disableGui && glfwWindowShouldClose(window)) { break; }
	}
	
	const uint32_t totalIterations = std::accumulate(totalIterationsPerLevel.begin(), totalIterationsPerLevel.end(), 0u);
	
	if (opts.verbose)
	{
		std::cout << "Total iterations: " << totalIterationsPerLevel[0] << ", " << totalIterationsPerLevel[1] << ", " << totalIterationsPerLevel[2] << ", " << totalIterationsPerLevel[3] << ", " << totalIterationsPerLevel[4] << ", " <<
		          " Total: " << totalIterations << '\n';
		std::cout << "Average iterations: "
		          << static_cast<double>(totalIterationsPerLevel[0])/static_cast<double>(totalFrames) << ", "
		          << static_cast<double>(totalIterationsPerLevel[1])/static_cast<double>(totalFrames) << ", "
		          << static_cast<double>(totalIterationsPerLevel[2])/static_cast<double>(totalFrames) << ", "
		          << static_cast<double>(totalIterationsPerLevel[3])/static_cast<double>(totalFrames) << ", "
		          << static_cast<double>(totalIterationsPerLevel[4])/static_cast<double>(totalFrames) << ", "
		          << " Total: " << static_cast<double>(totalIterations)/ static_cast<double>(totalFrames) << '\n';
		
		std::cout << "Max iterations: " << maxIterationsPerLevel[0] << ", " << maxIterationsPerLevel[1] << ", " << maxIterationsPerLevel[2] << ", " << maxIterationsPerLevel[3] << ", " << maxIterationsPerLevel[4] << '\n';
		
		std::cout << "Average elapsed time: " << (static_cast<double>(totalElapsedMicroseconds)/ static_cast<double>(totalFrames)) << " Max elapsed time: " << maxElapsedMicroseconds << '\n';
	}
	
	
	if (!opts.trajectoryFilePath.empty())
	{
		std::ofstream estimatedPosePathFile;
		estimatedPosePathFile.open(opts.trajectoryFilePath);
		
		estimatedPosePathFile << "# timestamp tx ty tz qx qy qz qw\n";
		
		std::stringstream ss;
		ss << std::fixed;
		
		for (const auto& pose: estimatedPosePath)
		{
			ss.str(""); // Reset
			ss.clear(); // Clear state flags.
			for (const auto& poseElement: pose)
			{
				ss << poseElement;
				if (&poseElement != pose.end())
				{ ss << " "; }
			}
			ss << '\n';
			
			estimatedPosePathFile << ss.str();
		}
		
		estimatedPosePathFile.close();
	}
	
	if (!opts.groundtruthFilePath.empty())
	{
		std::ofstream groundtruthPosePathFile;
		groundtruthPosePathFile.open(opts.groundtruthFilePath);
		
		groundtruthPosePathFile << "# timestamp tx ty tz qx qy qz qw\n";
		
		std::stringstream ss;
		ss << std::fixed;
		
		for (const auto& pose: groundtruthPosePath)
		{
			ss.str(""); // Reset
			ss.clear(); // Clear state flags.
			for (const auto& poseElement: pose)
			{
				ss << poseElement;
				if (&poseElement != pose.end()) { ss << " "; }
			}
			ss << '\n';
			
			groundtruthPosePathFile << ss.str();
		}
		
		groundtruthPosePathFile.close();
	}
	
	if (!opts.statisticsFilePath.empty())
	{
		std::ofstream statisticsFile;
		statisticsFile.open(opts.statisticsFilePath);
		
		statisticsFile << "Settings\n";
		statisticsFile << "Max iterations: " << alignmentSettings.maxIterations << '\n';
		statisticsFile << "Motion prior: " << alignmentSettings.motionPrior << '\n';
		statisticsFile << "xi convergence length: " << alignmentSettings.xiConvergenceLength << '\n';
		statisticsFile << "Initial step length: " << alignmentSettings.initialStepLength << '\n';
		statisticsFile << "Step length reduction factor: " << alignmentSettings.stepLengthReductionFactor << '\n';
		statisticsFile << "Min step length: " << alignmentSettings.minStepLength << '\n';
		
		
		statisticsFile << "\n\n";
		
		statisticsFile << "Total iterations\n";
		for (const auto& value: totalIterationsPerLevel)
		{
			statisticsFile << value << ' ';
		}
		statisticsFile << '\n';
		
		statisticsFile << "Average iterations\n";
		for (const auto& value: totalIterationsPerLevel)
		{
			statisticsFile << (static_cast<double>(value)/static_cast<double>(totalFrames)) << ' ';
		}
		statisticsFile << '\n';
		
		statisticsFile << "Max iterations\n";
		for (const auto& value: maxIterationsPerLevel)
		{
			statisticsFile << value << ' ';
		}
		statisticsFile << '\n';
		
		statisticsFile << "Average elapsed time: " << (static_cast<double>(totalElapsedMicroseconds)/ static_cast<double>(totalFrames)) << '\n';
		statisticsFile << "Max elapsed time: " << maxElapsedMicroseconds << '\n';
		
		statisticsFile.close();
	}
	
	
	return 0;
}