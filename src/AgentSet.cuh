#ifndef AGENT_SET_CUH
#define AGENT_SET_CUH

#include "Collection.cuh"

/**
 \class AgentSet
 \brief A collection to store a set of agents.
 This class provides the basic information of agents, their indices, if they are alive and if they are pregnant.
**/
class AgentSet : public Collection{
public:
	/**
	 * Allocate the memory of the agents.
	 * \param numAgents The quantity of agents.
	 * \param stream The <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a">cudaStream</a> that will handle the allocation. Default is the stream 0.
 	 **/
	virtual void alloc(const unsigned int numAgents, cudaStream_t stream);
	 /**
	 * Resize the set. If the new size is bigger than the current capacity, 
	 * this method will allocate a new block of memory and copy all data from 
	 * the old block of memory and then it will free the old block.
	 * \param oldSize The size of the collection.
	 * \param newSize The new size of the collection.
	 * \param stream The <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a">cudaStream</a> that will handle the resize. Default is the stream 0.
	 **/
	virtual void resize(const unsigned int oldSize, const unsigned int newSize, cudaStream_t stream);
	virtual void free();
	/**
	* Copy the data of a single agent from a given AgentSet. This function must be called only in a GPU kernel.
	* \param index1 The index of the agent that will receive the data.
	* \param in The input AgentSet containing the data to be copied.
	* \param index2 The input index where the agent's information is stored in input AgentSet.
	* @code
	this->indices[index1]  = in->indices[index2];
	this->alives[index1]   = in->alives[index2];
	this->pregnant[index1] = in->pregnants[index2];
	* @endcode
	**/
	virtual __device__ void copy(const unsigned int index1, const AgentSet *in, const unsigned int index2);
	/**
	* Rebirth a given agent in this set. This function must be called only in a GPU kernel.
	* Due to performance issues, the agents are not inserted or removed from the set, instead of this, they are setted as dead or alive.
	* To insert a new agent a dead agent need to be revived. This operation occurs internally but one can define which informations
	* should be restored.
	* \param index The index of the agent to be rebirth.
	* \param parent The index of its parent. The parent can be used to inherit information, for example the type of the new agent.
	@code
	this->alives[index]   = true;
	this->pregnant[index] = false;
	this->types[index]    = this->types[parent];
	@endcode
	**/
	virtual __device__ void rebirth(const unsigned int index, const unsigned int parent);
	/**
	* Set the state of a given agent as dead. This function must be called only in a GPU kernel.
	* \param index The index of the agent.
	**/
	__device__ void die(const unsigned int index);
	/**
	* Check if a given agent is dead. This function must be called only in a GPU kernel.
	* \param index The index of the agent.
	**/
	__device__ bool isDead(const unsigned int index) const;
	/**
	* Check if a given agent is alive. This function must be called only in a GPU kernel.
	* \param index The index of the agent.
	**/
	__device__ const bool& isAlive(const unsigned int index) const;
	/**
	* Set the state of a given agent as pregnant. This function must be called only in a GPU kernel.
	* \param index The index of the agent.
	**/
	__device__ void reproduce(const unsigned int index);
	/**
	* Check if a given agent is pregnant. This function must be called only in a GPU kernel.
	* \param index The index of the agent.
	**/
	__device__ const bool& isPregnant(const unsigned int index) const;

private:
	unsigned int *indices;
	bool *alives;
	bool *pregnants;
};

#endif
