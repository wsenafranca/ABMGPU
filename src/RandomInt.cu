#include "RandomInt.cuh"

void RandomInt::create(long seed = 0) {
	curandCreateGenerator(&p_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(p_gen, seed);
}
