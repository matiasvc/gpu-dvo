# gpu-dvo
CUDA Optimized DVO

Download datasets using the script `dataset/download.sh`

Example of launching the application:
```
gpu_dvo -i dataset/rgbd_dataset_freiburg3_long_office_household -p 535.4 539.2 320.1 247.6
```
Use the following intrinsic parameter settings for the dataset sequencens:
- freiburg1: `-p 517.3 516.5 318.6 255.3`
- freiburg2: `-p 520.9 521.0 325.1 249.7`
- freiburg3: `-p 535.4 539.2 320.1 247.6`

The application requires the following libraries to be installed:
- CUDA
- Nvidia VisionWorks
- Eigen
- Boost
- Sophus
- glfw

