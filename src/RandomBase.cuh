#ifndef RANDOM_BASE_CUH
#define RANDOM_BASE_CUH

#include <curand.h>
#include <curand_kernel.h>

/**
* \class RandomBase
* @brief Base class to generate random numbers for GPU.
* This class wraps the cuda RNG <a href="http://docs.nvidia.com/cuda/curand/index.html">cuRAND</a>.
**/
class RandomBase{
public:
	/**
	* Define the random type: interger, float, double...
	* \param seed The random seed used to internally create the RNG.
	**/
	virtual void create(long seed) = 0;
	/**
	* Generate random unsigned integers and store them in a given array. The numbers generated are between 0 and MAX_UNSIGNED_INT.
	* \param random An array of unsigned integers.
	* \param size The size of the array.
	**/
	void generate(unsigned int *random, const unsigned int size);
	/**
	* Generate random float numbers following the distribution Normal and store them in a given array.
	* \param random An array of float.
	* \param size The size of the array.
	* \param mean The mean of the normal distribution.
	* \param stdev The standard deviation of the normal distribution.
	**/
	void generateNormal(float *random, const unsigned int size, float mean, float stdev);
	/**
	* Generate random float numbers following the distribution Uniform and store them in a given array. The numbers generated are between 0 and 1.
	* \param random An array of float.
	* \param size The size of the array.
	**/
	void generateUniform(float *random, const unsigned int size);
	/**
	* Destoy the RNG.
	**/
	void destroy();
	
protected:
	/**
	* A pointer to the <a href="http://docs.nvidia.com/cuda/curand/index.html">cuRAND</a> generator.
	**/
	curandGenerator_t p_gen;
};

#endif
