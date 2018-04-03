#ifndef SOCIETY_CUH
#define SOCIETY_CUH

/**
* \class Society
* \brief A Society handles an @link AgentSet @endlink. 
* This class manages the memory of the set and provides to it a <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a">cudaStream</a>. 
**/

template<class T>
class Society {
public:
	Society() {}
	virtual ~Society() {}
	
	/**
	 * Allocate the memory of the internal @link AgentSet @endlink.
	 * \param numAgents The quantity of agents.
	 * \param capacity The maximum of elements that the @link AgentSet @endlink supports initially.
 	 **/
	virtual void alloc(const unsigned int numAgents, const unsigned int capacity) {
		cudaStreamCreate(&stream);
		
		this->capacity = capacity;
		this->numAgents = numAgents;
		
		agents.alloc(capacity, stream);
		buffer.alloc(capacity, stream);
	}
	
	/**
	 * Resize the set. If the new size is bigger than the current capacity, 
	 * this method will allocate a new block of memory and copy all data from 
	 * the old block of memory and then it will free the old block.
	 * \param oldSize The size of the Society.
	 * \param newSize The new size of the Society.
	 **/
	virtual void resize(const unsigned int oldSize, const unsigned int newSize) {
		agents.resize(oldSize,newSize,stream);
		buffer.resize(oldSize,newSize,stream);
	}
	
	/// Free the memory of the internal @link AgentSet @endlink and the internal <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a">cudaStream</a>.
	virtual void free() {
		agents.free();
		buffer.free();	
		cudaStreamDestroy(stream);
	}
	
	/// Synchronize the internal @link AgentSet @endlink informations. @see AgentSetKernels.
	void synchronize() {
		unsigned int oldCapacity = capacity;
		synchronizeAgents(&agents, &buffer, &numAgents, &capacity, stream);
		std::swap(agents, buffer);
		
		if(capacity > oldCapacity) {
			resize(oldCapacity, capacity);
		}
	}

private:
	unsigned int numAgents;
	unsigned int capacity;
	T agents, buffer;
	cudaStream_t stream;
};

#endif
