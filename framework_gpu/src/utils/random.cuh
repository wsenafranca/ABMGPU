#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <curand_kernel.h>
#include "utils.cuh"

__device__ curandState* globalState;

__device__ float cuRandom(uint id) {
    curandState localState = globalState[id];
    float rndval = curand_uniform( &localState );
    globalState[id] = localState;
    return rndval;
}

template<class T>
__device__ T cuRand(uint id, T minValue, T maxValue) {
    curandState localState = globalState[id];
    float rndval = curand_uniform( &localState );
    globalState[id] = localState;
    rndval = fma(rndval, maxValue - minValue, minValue);
    return truncf(rndval);
}

template<class T>
__device__ T cuRand(uint id, T maxValue) {
    return cuRand(id, (T)0, maxValue);
}

__global__ void registerRandState(curandState *state) {
    globalState = state;
}

__global__ void initRandomKernel(curandState *state, size_t size, long seed) {
    uint idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < size)
        curand_init ( seed, idx, 0, &state[idx] );
}

class Random {
public:
    Random(long seed, uint elements) {
		cudaMalloc(&state, sizeof(curandState)*elements);
		CHECK_ERROR
		uint blocks = BLOCKS(elements);
		registerRandState<<<1,1>>>(state);
		initRandomKernel<<<blocks, THREADS>>>(state, elements, seed);
		CHECK_ERROR
	}
    ~Random() {
		cudaFree(state);
	}
private:
    curandState *state;
};

#endif

