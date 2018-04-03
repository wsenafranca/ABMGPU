#ifndef RANDOM_FLOAT_CUH
#define RANDOM_FLOAT_CUH

#include "RandomBase.cuh"

/**
* \class RandomFloat
* \brief Class to generate random float numbers for GPU.
**/
class RandomFloat : public RandomBase {
public:
	/**
	* Define a float random type.
	* \param seed The random seed used to internally create the RNG.
	**/
	void create(long seed);
};

#endif
