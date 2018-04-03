#include <ABMGPU.h>

// Define a set of agents (boids) in a struct of arrays style
struct Boids : public AgentSet{
	void alloc(const unsigned int numBoids, cudaStream_t stream) override {
		AgentSet::alloc(numBoids, stream);
		cudaMalloc(&position, sizeof(float3)*numBoids);
		cudaMalloc(&heading, sizeof(float3)*numBoids);
		cudaMemset(heading, 0, sizeof(float3)*numBoids);
	}

	void free() {
		AgentSet::free();
		cudaFree(position);
		cudaFree(heading);
	}

	float3 *position;
	float3 *heading;
};

// Store the data from the last sync state
struct Past {
	float3 *position;
	float3 *heading;
};

// Store neighborhood bounds
struct Bounds {
	unsigned int *begin;
	unsigned int *end;
};

// normalize a vector3f
__device__ void normalize(float3 *position) {
	float m = sqrtf((position->x)*(position->x) + (position->y)*(position->y) + (position->z)*(position->z));
	if(m != 0) {
		position->x = position->x/m;
		position->y = position->y/m;
		position->z = position->z/m;
	}
}

// calc the distance^2
__device__ float dist2(const float3 &v1, const float3 &v2) {
	return ((v1.x-v2.x)*(v1.x-v2.x)) + ((v1.y-v2.y)*(v1.y-v2.y)) + ((v1.z-v2.z)*(v1.z-v2.z));
}

// transform a real position into a grid position
__device__ int discretize(float x, int dim, int d) {
	return (x + (dim/2))/d;
}

// map a boid position into the space coordinate
struct HashBoid {
	HashBoid(float3 *position, const int3 &dimension, const unsigned int &neighborhood) : 
		position(position), dimension(dimension), neighborhood(neighborhood) {}
	__device__ int operator()(const unsigned int index) const {
		int nx = discretize(position[index].x, dimension.x, neighborhood);
		int nz = discretize(position[index].z, dimension.z, neighborhood);
		return nz * (dimension.x/neighborhood) + nx;
	}
	float3 *position;
	const int3 &dimension;
	const unsigned int &neighborhood;
};

// set boids position
__global__ void placement(Boids boids, const float *random, const unsigned int numBoids, const int3 dimension) {
	const unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < numBoids) {
		// get the position in random table
		const float x = random[i*2  ]*(dimension.x-1.0f)-(dimension.x/2.0f);
		const float y = random[i*2+1]*(dimension.y-1.0f)-(dimension.y/2.0f);
		const float z = random[i*2+2]*(dimension.z-1.0f)-(dimension.z/2.0f);
		// clamp position
		boids.position[i].x = max(-dimension.x/2.0f, min(x, (dimension.x-1)/2.0f));
		boids.position[i].y = max(-dimension.y/2.0f, min(y, (dimension.y-1)/2.0f));
		boids.position[i].z = max(-dimension.z/2.0f, min(z, (dimension.z-1)/2.0f));
	}
}

// perform the flocking
__global__ void flocker(Boids boids, const Past past, const Bounds bounds, const float *random, 
	const unsigned int numBoids, const int3 dimension, const int neighborhood) 
{
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid < numBoids) {
		// get a boid in the set. Optimizing the access using the indices
		const int index = boids.indices[tid]; // try index = boids[tid] to compare

		// copy information from global memory to local cache
		float3 position = past.position[index];
		float3 heading = past.heading[index];

		// transform boid's position into 2d grid
		int gridX = discretize(position.x, dimension.x, neighborhood);
		int gridY = discretize(position.z, dimension.z, neighborhood);

		// define the vectors used to compute next position of a boid
		float3 alignment  = make_float3(0,0,0);
		float3 cohesion   = make_float3(0,0,0);
		float3 separation = make_float3(0,0,0);
		// search for boids in view field (neighborhood)
		unsigned int countNeighbor = 0;
		#pragma unroll // unroll loops to impove performance
		for(int i = -1; i <= 1; i++) {
			#pragma unroll
			for(int j = -1; j <=1; j++) {
				// test if position is inside the grid
				if(gridX+j >= 0 && gridX+j < dimension.x / neighborhood && 
					gridY+i >= 0 && gridY+i < dimension.z / neighborhood) 
				{
					int neighborhoodOnGrid = (i+gridY) * (dimension.x/neighborhood) + (j+gridX);
					// look for boids in the bounds
					for(int k = bounds.begin[neighborhoodOnGrid]; k < bounds.end[neighborhoodOnGrid]; k++) {
						// get information from neighbors
						float3 neighPosition = past.position[k];
						float3 neighVelocity = past.heading[k];
						// check if the neigh is visible
						if(boids.indices[k] != index && dist2(position, neighPosition) < neighborhood*neighborhood) {
							countNeighbor++;

							alignment.x += neighVelocity.x;
							alignment.y += neighVelocity.y;
							alignment.z += neighVelocity.z;

							cohesion.x += neighPosition.x;
							cohesion.y += neighPosition.y;
							cohesion.z += neighPosition.z;

							separation.x += neighPosition.x - position.x;
							separation.y += neighPosition.y - position.y;
							separation.z += neighPosition.z - position.z;
						}
					}
				}
			}
		}

		if(countNeighbor > 0) {
			alignment.x /= countNeighbor;
			alignment.y /= countNeighbor;
			alignment.z /= countNeighbor;
			normalize(&alignment);

			cohesion.x /= countNeighbor;
			cohesion.y /= countNeighbor;
			cohesion.z /= countNeighbor;
			cohesion.x -= position.x;
			cohesion.y -= position.y;
			cohesion.z -= position.z;
			normalize(&cohesion);

			separation.x /= countNeighbor;
			separation.y /= countNeighbor;
			separation.z /= countNeighbor;
			separation.x = -separation.x;
			separation.y = -separation.y;
			separation.z = -separation.z;
			normalize(&separation);
		}

		// random factor
		float3 r;
		r.x = fma(random[index*2  ], 2.0f, -1.0f); // float multiply add in one call
		r.y = fma(random[index*2+1], 2.0f, -1.0f);
		r.z = fma(random[index*2+2], 2.0f, -1.0f);

		// calculate direction
		heading.x += alignment.x + separation.x + cohesion.x + r.x;
		heading.y += alignment.y + separation.y + cohesion.y + r.y;
		heading.z += alignment.z + separation.z + cohesion.z + r.z;
		normalize(&heading);

		// perform next step

		// invert heading if boids is out of bound
		heading.x = position.x+heading.x < -dimension.x/2.0f || position.x+heading.x >= dimension.x/2.0f ? -heading.x : heading.x;
		heading.y = position.y+heading.y < -dimension.y/2.0f || position.y+heading.y >= dimension.y/2.0f ? -heading.y : heading.y;
		heading.z = position.z+heading.z < -dimension.z/2.0f || position.z+heading.z >= dimension.z/2.0f ? -heading.z : heading.z;
		position.x += heading.x;
		position.y += heading.y;
		position.z += heading.z;

		// save current state
		boids.position[index] = position;
		boids.heading[index] = heading;
	}
}

class FlockingBoids : public Simulation
{
public:
	FlockingBoids(unsigned int numBoids, int3 dimension, unsigned int neighborhood) : 
		numBoids(numBoids), dimension(dimension), neighborhood(neighborhood) {}
	
	void init() override {
		// allocate memory in GPU
		boids.alloc(numBoids, 0);
		grid = make_int2(dimension.x, dimension.z);
		spatialNeighborhood.alloc(numBoids, grid, neighborhood);
		cudaMalloc(&past.position, sizeof(float3)*numBoids);
		cudaMalloc(&past.heading, sizeof(float3)*numBoids);
		cudaMalloc(&random, sizeof(float)*numBoids*3);

		rng.create(0); // seed 0
		rng.generateUniform(random, numBoids*3); // generate random numbers folloing the uniform distribution

		// place boids in the space
		int minGridSize,blockSize, gridSize;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, placement, 0, 0);
		gridSize = (numBoids + blockSize - 1) / blockSize;
		placement<<<gridSize, blockSize>>>(boids, random, numBoids, dimension);
	}

	void communicate() override {
		// organize spatially the boids in the 2d space.
		spatialNeighborhood.index(boids.indices, numBoids, grid, neighborhood, HashBoid(boids.position, dimension, neighborhood));
		bounds.begin = spatialNeighborhood.begins;
		bounds.end = spatialNeighborhood.ends;
	}

	void step(int time) override {
		// generate random numbers
		rng.generateUniform(random, numBoids*3);

		// perform the flocking
		int minGridSize,blockSize, gridSize;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, flocker, 0, 0);
		gridSize = (numBoids + blockSize - 1) / blockSize;
		flocker<<<gridSize, blockSize>>>(boids, past, bounds, random, numBoids, dimension, neighborhood);
	}

	void synchronize() override {
		// synchronize past information with current state
		cudaMemcpy(past.position, boids.position, sizeof(float3)*numBoids, cudaMemcpyDeviceToDevice);
		cudaMemcpy(past.heading, boids.heading, sizeof(float3)*numBoids, cudaMemcpyDeviceToDevice);
	}

	void finalize() override {
		// cleanup memory
		boids.free();
		spatialNeighborhood.free();
		rng.destroy();
		cudaFree(past.position);
		cudaFree(past.heading);
		cudaFree(random);
	}

private:
	unsigned int numBoids;
	int3 dimension;
	int2 grid;
	unsigned int neighborhood;
	Boids boids;
	Past past;
	RandomFloat rng;
	float *random;
	SpatialNeighborhood spatialNeighborhood;
	Bounds bounds;
};

int main(int argc, char **argv) {
	int numAgents = 50000;
	int3 dim = make_int3(500, 500, 500);
	int neighborhood = 10;
	int device = 0;
	
	if(argc > 1)
		sscanf(argv[1], "%d", &numAgents);
	if(argc > 2) {
		sscanf(argv[2], "%d", &dim.x);
		sscanf(argv[2], "%d", &dim.y);
		sscanf(argv[2], "%d", &dim.z);
	}
	if(argc > 3)
		sscanf(argv[3], "%d", &neighborhood);
	if(argc > 4)
		sscanf(argv[4], "%d", &device);

	cudaSetDevice(device);

	FlockingBoids sim(numAgents, dim, neighborhood);
	sim.execute(100);
}
