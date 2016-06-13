#==============================================
# Demos
#==============================================

BIN = bin

DEMOS_CUDA_DIR = demos/cuda
DEMOS_CPP_DIR = demos/cpp

PREDATOR_PREY_GPU = $(DEMOS_CUDA_DIR)/predator_prey/predator_prey.cu
PREDATOR_PREY_CPU = $(DEMOS_CPP_DIR)/predator_prey/predator_prey.cpp

FLOCKER_GPU = $(DEMOS_CUDA_DIR)/flocking_boid/flocker.cu
FLOCKER_CPU = $(DEMOS_CPP_DIR)/flocking_boid/flocker.cpp

SCHEDULER_GPU = $(DEMOS_CUDA_DIR)/scheduler/scheduler.cu

GAME_OF_LIFE_GPU = $(DEMOS_CUDA_DIR)/game_of_life/game_of_life.cu

#==============================================

FRAMEWORK_DIR = framework_gpu

SRC_DIR = $(FRAMEWORK_DIR)/src

LIBRARIES = -lcurand

COMPILER = --std=c++11

FLAGS = -default-stream per-thread

INCLUDES = -I$(SRC_DIR) -I$(FRAMEWORK_DIR)

all: mkbin preys_gpu boids_gpu preys_cpu boids_cpu

debug: mkbin debug_preys_gpu debug_boids_gpu

mkbin:
	@mkdir -p $(BIN)

scheduler_gpu:
	@nvcc ${COMPILER} ${INCLUDES} ${LIBRARIES} ${FLAGS} $(SCHEDULER_GPU) -o $(BIN)/$@

gameoflife_gpu:
	@nvcc ${COMPILER} ${INCLUDES} ${LIBRARIES} ${FLAGS} $(GAME_OF_LIFE_GPU) -o $(BIN)/$@ -lineinfo -g

preys_gpu: 
	@nvcc ${COMPILER} ${INCLUDES} ${LIBRARIES} ${FLAGS} $(PREDATOR_PREY_GPU) -o $(BIN)/$@
	
boids_gpu: 
	@nvcc ${COMPILER} ${INCLUDES} ${LIBRARIES} ${FLAGS} $(FLOCKER_GPU) -o $(BIN)/$@
	
debug_preys_gpu: 
	@nvcc ${COMPILER} ${INCLUDES} ${LIBRARIES} ${FLAGS} $(PREDATOR_PREY_GPU) -o $(BIN)/preys_gpu -lineinfo -g
	
debug_boids_gpu: 
	@nvcc ${COMPILER} ${INCLUDES} ${LIBRARIES} ${FLAGS} $(FLOCKER_GPU) -o $(BIN)/boids_gpu -lineinfo -g
	
preys_cpu:
	@g++ $(PREDATOR_PREY_CPU) -o $(BIN)/$@

boids_cpu: 
	@g++ $(FLOCKER_CPU) -o $(BIN)/$@
