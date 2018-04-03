cmake_minimum_required(VERSION 3.9)
project(Demos)

find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=compute_30 -code=sm_30 -std=c++11")

find_path(ABMGPU_DIR REQUIRED)
find_library(ABMGPU_LIB REQUIRED)
include_directories(${ABMGPU_DIR})

option(GAME_OF_LIFE "Build Game of Life?" ON)
if(GAME_OF_LIFE)
	cuda_add_executable(GameOfLife GameOfLife.cu)
	target_link_libraries(GameOfLife ${ABMGPU_LIB})
endif()

option(FLOCKING_BOIDS "Build Flocking Boids?" ON)
if(FLOCKING_BOIDS)
	cuda_add_executable(FlockingBoids FlockingBoids.cu)
	target_link_libraries(FlockingBoids ${ABMGPU_LIB})
endif()

