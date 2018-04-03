#ifndef RANDOM_INT_CUH
#define RANDOM_INT_CUH

#include "RandomBase.cuh"

/**
* \class RandomInt
* \brief Class to generate random unsigned int numbers for GPU.
**/
class RandomInt : public RandomBase {
public:
	/**
	* Define an integer random type.
	* \param seed The random seed used to internally create the RNG.
	**/
	void create(long seed);
};

#endif
