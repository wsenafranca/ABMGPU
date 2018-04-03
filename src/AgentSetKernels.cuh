#ifndef AGENT_SET_KERNELS_CUH
#define AGENT_SET_KERNELS_CUH

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>

/**
* The kernel used to reorder the agents of a given AgentSet.
* This kernel reorder an input AgentSet into a output AgentSet.
* \param out An AgentSet to store the agents reordered.
* \param in An AgentSet with the agents to be reordered.
* \param numAgent The quantity of agents to be reordered.
**/
template<class A>
__global__ void reorder_kernel(A out, const A in, const unsigned int numAgents) {
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid < numAgents) {
		out.copy(tid, &in, in.indices[tid]);
	}
}

/**
* The kernel to rebirth dead agents.
* This kernel uses a hash to determine where are the dead agents that will be used to insert the children of the pregnant agents.
* \param agents The AgentSet.
* \param hash A hash table that maps a given pregnant agent into a dead agent.
* \param numAgents The quantity of agents in the set.
**/
template<class A>
__global__ void rebirth_kernel(A agents, const unsigned int *hash, const unsigned int numAgents) {
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid < numAgents) {
		const unsigned int agent = agents.indices[tid];
		if(agents.isPregnant(agent)) {
			const unsigned int child = hash[agent]+numAgents;
			agents.rebirth(child, agent);
			agents.pregnants[agent] = false;
		}
	}
}

/**
* \class HashByKey
* \brief Helper class to organize agents in GPU using a key-value hash.
**/
template<class T>
class HashByKey{
public:
	/**
	* Create a key-value hash struct.
	* \param values The values to be accessed by the keys. It will generate the hash table.
	**/
	HashByKey(const T *values) : values(values) {}
	
	/**
	* The operator that maps a given key in some value.
	* \param key A key that will access some value.
	**/
	__device__ unsigned int operator()(const unsigned int key) {
		return values[key];
	}
	
private:
	const T *values;
};

__global__ void testAlives(bool *alives, unsigned int *indices, unsigned int numAgents) {
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid < numAgents) {
		assert(indices[alives[tid]]);
	}
}

/**
* A function to synchronize the agents, reordering the alive agents in the begin of the set and inserting the new agents where are the dead agents.
* \param out An AgentSet to store the agents synchronized.
* \param in An AgentSet with the agents to be synchronized.
* \param pNumAgent A pointer to the quantity of agents to be synchronized. The quantity of agents may change after synchronize.
* \param pCapacity A pointer to the capacity of agents in the set. The capacity of agents may change after synchronize.
* \param stream The cudaStream that will handle the synchronize. Default is the stream 0.
**/
template<class T>
void synchronizeAgents(T *in, T *out, unsigned int *pNumAgents, unsigned int *pCapacity, cudaStream_t stream=0) {
	const unsigned int numAgents = *pNumAgents;
	if(numAgents == 0) return;
	
	int minGridSize,blockSize, gridSize;
	T &past = *in;
	T &agents = *out;
	
	unsigned int *pend = thrust::partition(thrust::device, past.indices, past.indices+numAgents, HashByKey<bool>(past.alives));	
	unsigned int numAlives = pend - past.indices;
	
	if(numAlives == 0) {
		*pNumAgents = 0;
		return;
	}
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reorder_kernel<T>, 0, 0);
	gridSize = (numAgents + blockSize - 1) / blockSize;
	reorder_kernel<T><<<gridSize, blockSize, 0, stream>>>(agents, past, numAgents);
	
	unsigned int *hash;
	cudaMalloc(&hash, sizeof(unsigned int)*numAlives);
	
	cudaStreamSynchronize(0);
	thrust::transform(thrust::cuda::par.on(stream), agents.indices, agents.indices+numAlives, hash, HashByKey<bool>(agents.pregnants));
	cudaStreamSynchronize(stream);
	
	unsigned int numNewAgents = thrust::reduce(thrust::device, hash, hash+numAlives);
	cudaStreamSynchronize(0);
	
	thrust::exclusive_scan(thrust::cuda::par.on(stream), hash, hash+numAlives, hash);
	cudaStreamSynchronize(stream);
	
	if(numNewAgents > 0) {
		if(numNewAgents+numAlives > *pCapacity) {
			unsigned int oldCapacity = *pCapacity;
			*pCapacity = (numNewAgents+numAlives)*2;
			agents.resize(oldCapacity, *pCapacity, stream);
			past.resize(oldCapacity, *pCapacity, stream);
			cudaStreamSynchronize(stream);
		}
		
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rebirth_kernel<T>, 0, 0);
		gridSize = (numAlives + blockSize - 1) / blockSize;
		rebirth_kernel<T><<<gridSize, blockSize, 0, stream>>>(agents, hash, numAlives);
		cudaStreamSynchronize(stream);
	}
	
	*pNumAgents = numNewAgents + numAlives;

	cudaFree(hash);
}

#endif
