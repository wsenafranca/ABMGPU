#include "AgentSet.cuh"

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

void AgentSet::alloc(const unsigned int numAgents, cudaStream_t stream = 0) {
	cudaMalloc(&indices, sizeof(unsigned int)*numAgents);
	cudaMalloc(&alives, sizeof(bool)*numAgents);
	cudaMalloc(&pregnants, sizeof(bool)*numAgents);
	
	thrust::sequence(thrust::cuda::par.on(stream), indices, indices+numAgents);
	thrust::fill(thrust::cuda::par.on(stream), alives, alives+numAgents, true);
	thrust::fill(thrust::cuda::par.on(stream), pregnants, pregnants+numAgents, false);
}

void AgentSet::resize(const unsigned int oldSize, const unsigned int newSize, cudaStream_t stream=0) {
	
	unsigned int *h_indices;
	bool *h_alives;
	bool *h_pregnants;
	
	cudaMallocHost(&h_indices, sizeof(unsigned int)*oldSize);
	cudaMallocHost(&h_alives, sizeof(bool)*oldSize);
	cudaMallocHost(&h_pregnants, sizeof(bool)*oldSize);
	
	cudaMemcpyAsync(h_indices, indices, sizeof(unsigned int)*oldSize, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(h_alives, alives, sizeof(bool)*oldSize, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(h_pregnants, pregnants, sizeof(bool)*oldSize, cudaMemcpyDeviceToHost, stream);
	
	cudaFree(indices);
	cudaFree(alives);
	cudaFree(pregnants);
	cudaMalloc(&indices, sizeof(unsigned int)*newSize);
	cudaMalloc(&alives, sizeof(bool)*newSize);
	cudaMalloc(&pregnants, sizeof(bool)*newSize);
	
	cudaMemcpyAsync(indices, h_indices, sizeof(unsigned int)*oldSize, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(alives, h_alives, sizeof(bool)*oldSize, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(pregnants, h_pregnants, sizeof(bool)*oldSize, cudaMemcpyHostToDevice, stream);
	
	cudaFreeHost(h_indices);
	cudaFreeHost(h_alives);
	cudaFreeHost(h_pregnants);
	
	cudaStreamSynchronize(stream);
}

void AgentSet::free() {
	cudaFree(indices);
	cudaFree(alives);
	cudaFree(pregnants);
}

__device__ void AgentSet::copy(const unsigned int index1, const AgentSet *in, const unsigned int index2) {
	indices[index1] = index1;
	alives[index1] = in->alives[index2];
	pregnants[index1] = in->pregnants[index2];
}

__device__ void AgentSet::rebirth(const unsigned int index, const unsigned int parent) {
	indices[index] = index;
	alives[index] = true;
	pregnants[index] = false;
}

__device__ void AgentSet::die(const unsigned int index) {
	alives[index] = false;
}

__device__ bool AgentSet::isDead(const unsigned int index) const {
	return !alives[index];
}

__device__ const bool& AgentSet::isAlive(const unsigned int index) const {
	return alives[index];
}

__device__ void AgentSet::reproduce(const unsigned int index) {
	pregnants[index] = true;
}

__device__ const bool& AgentSet::isPregnant(const unsigned int index) const {
	return pregnants[index];
}
