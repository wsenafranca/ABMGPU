#include "RandomBase.cuh"

void RandomBase::generate(unsigned int *random, const unsigned int size) {
	curandGenerate(p_gen, random, size);
}

void RandomBase::generateNormal(float *random, const unsigned int size, float mean=0, float stdev=1) {
	curandGenerateNormal(p_gen, random, size, mean, stdev);
}

void RandomBase::generateUniform(float *random, const unsigned int size) {
	curandGenerateUniform(p_gen, random, size);
}

void RandomBase::destroy() {
	curandDestroyGenerator(p_gen);
}
