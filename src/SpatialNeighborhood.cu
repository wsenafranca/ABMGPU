#include "SpatialNeighborhood.cuh"

__global__ void spatial_index_kernel(unsigned int *begins, unsigned int *ends, const unsigned int *hash, const unsigned int numAgents) {
    const int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid < numAgents) {
        const int index = hash[tid];
        
        if(tid == 0 || index != hash[tid-1])
            begins[index] = tid;
        if(tid==numAgents-1 || index != hash[tid+1])
            ends[index] = tid+1;
    }
}

void SpatialNeighborhood::alloc(const unsigned int numAgents, const int2 dimension, const unsigned int neighborhood) {
	unsigned int nx = dimension.x/neighborhood;
	unsigned int ny = dimension.y/neighborhood;
	
	cudaMalloc(&begins, sizeof(unsigned int)*nx*ny);
	cudaMalloc(&ends, sizeof(unsigned int)*nx*ny);
	
	cudaChannelFormatDesc channelDesc;
	
	channelDesc = cudaCreateChannelDesc<unsigned int>();
    cudaMallocArray(&beginsArray, &channelDesc, nx, ny, cudaArraySurfaceLoadStore);
    cudaBindTextureToArray(beginsTex, beginsArray, channelDesc);
    beginsTex.addressMode[0] = cudaAddressModeBorder;
	beginsTex.addressMode[1] = cudaAddressModeBorder;
    
    channelDesc = cudaCreateChannelDesc<unsigned int>();
    cudaMallocArray(&endsArray, &channelDesc, nx, ny, cudaArraySurfaceLoadStore);
    cudaBindTextureToArray(endsTex, endsArray, channelDesc);
    endsTex.addressMode[0] = cudaAddressModeBorder;
	endsTex.addressMode[1] = cudaAddressModeBorder;
}

void SpatialNeighborhood::free() {
	cudaFree(begins);
	cudaFree(ends);
	
	cudaFreeArray(beginsArray);
    cudaFreeArray(endsArray);
}
