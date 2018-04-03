#include "ABMGPU.h"
#include <cstdio>

// Game of life kernel.
__global__ void life(bool *out, const bool *in, size_t width, size_t height) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid < width*height) {
		bool cell = in[tid];
		int x = tid % width;
		int y = tid / width;
		int neighborhood = 0;
		for(int neigh_y = y-1; neigh_y <= y+1; neigh_y++) {
			for(int neigh_x = x-1; neigh_x <= x+1; neigh_x++) {
				neighborhood += neigh_x >= 0 && neigh_x < width && neigh_y >= 0 && neigh_y < height && in[neigh_y*width + neigh_x];
			}
		}

		out[tid] = (cell && (neighborhood == 2 || neighborhood == 3)) || (!cell && neighborhood == 3);
	}
}

// Create the simulation
class GameOfLife : public Simulation
{
public:
	GameOfLife(size_t width, size_t height) : width(width), height(height) {}

	// allocate data in GPU
	void init() override {
		// create the space
		cudaMalloc(&space, sizeof(bool)*width*height);
		// create a copy of the space to store the last state of the simulation
		cudaMalloc(&past, sizeof(bool)*width*height);

		// initialize the space with alive and empty cells
		bool *alives = new bool[width*height];
		for(size_t i = 0; i < width*height; i++)
			alives[i] = rand() % 2;
		// copy the memory allocated in CPU to GPU. Insert the data in the past memory
		cudaMemcpy(past, alives, sizeof(bool)*width*height, cudaMemcpyHostToDevice);
		// free CPU memory
		delete [] alives;
	}

	void step(int time) override {
		// find the max potential block size for the kernel "life"
		int minGridSize, blockSize, gridSize;
		size_t size = width*height;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, life, 0, 0);
		gridSize = (size + blockSize - 1) / blockSize;

		// perform the game of life
		life<<<gridSize, blockSize>>>(space, past, width, height);
	}

	void synchronize() override {
		// store the current state in the last state
		cudaMemcpy(past, space, sizeof(bool)*width*height, cudaMemcpyDeviceToDevice);
	}

	void finalize() override {
		// cleanup the GPU
		cudaFree(space);
		cudaFree(past);
	}

private:
	// the dimension of the space
	size_t width, height;
	// pointer to space and past memories
	bool *space, *past;
};

int main(int argc, char **argv) {
	size_t size = 1000;
	int device = 0;
	
	if(argc > 1) {
		sscanf(argv[1], "%ld", &size);
	}
	if(argc > 2) {
		sscanf(argv[2], "%d", &device);
	}
	
	// create the simulation
	GameOfLife sim(size, size);
	// execute the simulation for 100 iteration
	sim.execute(100);
	return 0;
}
