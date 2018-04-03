#ifndef COLLECTION_CUH
#define COLLECTION_CUH

/**
 * \class Collection
 * \brief An interface to define a collection of objects as a Struct of Array (SAO) that fits better in GPU.
*/

class Collection{
public:
	/**
	* Allocate the memory for a given numElements in GPU.
	* \param numElements The quantity of elements to be allocate.
	* \param stream The <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a">cudaStream</a> that will handle the allocation. Default is the stream 0.
	*/
	virtual void alloc(const unsigned int numElements, cudaStream_t stream) = 0;
	/// Free the memory of this collection. Note: This method is called internally.
	virtual void free() = 0;
	/**
	* Resize this collection for a new size.
	* \param oldSize The size of the collection.
	* \param newSize The new size of the collection.
	* \param stream The <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a">cudaStream</a> that will handle the resize. Default is the stream 0.
	*/
	virtual void resize(const unsigned int oldSize, const unsigned int newSize, cudaStream_t stream) = 0;
};

#endif

