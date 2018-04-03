cmake_minimum_required(VERSION 3.9)
project(Demos)

find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=compute_30 -code=sm_30 -std=c++11")

set(DEMOS_SRC
	GameOfLife.cu
)

cuda_add_executable(Demos ${DEMOS_SRC})

find_path(ABMGPU_DIR REQUIRED)
find_library(ABMGPU_LIB REQUIRED)
include_directories(${ABMGPU_DIR})
target_link_libraries(Demos ${ABMGPU_LIB})