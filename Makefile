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

#==============================================

SRC_DIR = framework_gpu

all: mkbin preys_gpu boids_gpu preys_cpu boids_cpu

debug: mkbin d_preys_gpu d_boids_gpu

mkbin:
	@mkdir -p $(BIN)

preys_gpu: 
	@nvcc -I $(SRC_DIR) $(PREDATOR_PREY_GPU) -o $(BIN)/$@

preys_cpu:
	@g++ $(PREDATOR_PREY_CPU) -o $(BIN)/$@
	
boids_gpu: 
	@nvcc -I $(SRC_DIR) -lcurand $(FLOCKER_GPU) -o $(BIN)/$@
	
boids_cpu: 
	@g++ $(FLOCKER_CPU) -o $(BIN)/$@
	
	
d_preys_gpu: 
	@nvcc -I $(SRC_DIR) -lcurand $(PREDATOR_PREY_GPU) -o $(BIN)/preys_gpu -lineinfo -g
	
d_boids_gpu: 
	@nvcc -I $(SRC_DIR) $(FLOCKER_GPU) -o $(BIN)/boids_gpu -lineinfo -g
	

