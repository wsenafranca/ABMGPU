#ifndef SPATIAL_NEIGHBORHOOD_CUH
#define SPATIAL_NEIGHBORHOOD_CUH

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> beginsTex, endsTex;

__global__ void spatial_index_kernel(unsigned int *begins, unsigned int *ends, const unsigned int *hash, const unsigned int numAgents);

/**
* \class SpatialNeighborhood
* \brief A class to index the agents into the space.
* This class splits agents into a grid using texture memory.
**/
class SpatialNeighborhood {
public:
	/**
	* Allocate the neighborhood in GPU.
	* \param numAgents The quantity of agents.
	* \param dimension The dimension of the space.
	* \param neighborhood The size of the neighborhood.
	**/
	void alloc(const unsigned int numAgents, const int2 dimension, const unsigned int neighborhood);
	/// Free the memory of the neighborhood. Note: This method is called internally.
	void free();
	
	/**
	* Reorder a given array of indices grouping the spatially near agents in the memory.
	* This function splits the agents into subregions of size equals to dimension / neighborhood.
	* \param indices The array of indices of an @link AgentSet @endlink.
	* \param numAgents The quantity of agents.
	* \param dimension The dimension of the space.
	* \param neighborhood The size of the neighborhood.
	* \param hash A hash function that maps a given agent index into a space coordinate.
	* \param stream The <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a">cudaStream</a> that will handle the index operation. Default is the stream 0.
	**/
	template<class H>
	void index(unsigned int *indices, const unsigned int numAgents, const int2 dimension, const unsigned int neighborhood, H hash, cudaStream_t stream = 0) {
		int nx = dimension.x/neighborhood;
		int ny = dimension.y/neighborhood;
		cudaMemset(begins, 0, sizeof(unsigned int)*nx*ny);
		cudaMemset(ends, 0, sizeof(unsigned int)*nx*ny);
		
		if(numAgents == 0) return;
		
		unsigned int *keys;
		cudaMalloc(&keys, sizeof(unsigned int)*numAgents);
		
		int minGridSize, blockSize, gridSize;
		
		thrust::transform(thrust::cuda::par.on(stream), indices, indices+numAgents, keys, hash);
		thrust::stable_sort_by_key(thrust::cuda::par.on(stream), keys, keys+numAgents, indices);
		
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, spatial_index_kernel, 0, 0);
		gridSize = (numAgents + blockSize - 1) / blockSize;
		spatial_index_kernel<<<gridSize, blockSize, 0, stream>>>(begins, ends, keys, numAgents);
		
		cudaMemcpyToArrayAsync(beginsArray,0, 0, begins, sizeof(unsigned int)*nx*ny,cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyToArrayAsync(endsArray,0, 0, ends, sizeof(unsigned int)*nx*ny,cudaMemcpyDeviceToDevice, stream);
        
        cudaFree(keys);
	}
	
	/**
	* Reorder a given array of indices placing the agents spatially near sequentially in the memory.
	* \param indices The array of indices of an @link AgentSet @endlink.
	* \param numAgents The quantity of agents.
	* \param hash A hash function that maps a given agent index into a space coordinate.
	* \param stream The <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a">cudaStream</a> that will handle the index operation. Default is the stream 0.
	**/
	template<class H>
	void index(unsigned int *indices, const unsigned int numAgents, H hash, cudaStream_t stream = 0) {
		if(numAgents == 0) return;
		
		unsigned int *keys;
		cudaMalloc(&keys, sizeof(unsigned int)*numAgents);
		
		thrust::transform(thrust::cuda::par.on(stream), indices, indices+numAgents, keys, hash);
		thrust::stable_sort_by_key(thrust::cuda::par.on(stream), keys, keys+numAgents, indices);
		
		cudaFree(keys);
	}

	unsigned int *begins;
	unsigned int *ends;
	
	cudaArray *beginsArray, *endsArray;
};

#endif
