#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <curand_kernel.h>

__device__ uint RANDOM_ELEMENTS;

#include <curand.h>
__device__ float *g_random;

__device__ float cuRandom(uint id) {
    return g_random[id];
}

template<class T>
__device__ T cuRand(uint id, T minValue, T maxValue) {
    float rndval = g_random[id < RANDOM_ELEMENTS ? id : id % RANDOM_ELEMENTS];
    rndval = rndval * (maxValue - minValue) + minValue;
    return truncf(rndval);
}

template<class T>
__device__ T cuRand(uint id, T maxValue) {
    return cuRand(id, (T)0, maxValue);
}

__global__ void registerRandonObj(float *randomObj, uint size) {
    g_random = randomObj;
    RANDOM_ELEMENTS = size;
}

class Random{
public:
    Random(long seed, uint size) {
        this->size = size;
        cudaMalloc(&d_random, sizeof(float)*size);
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        registerRandonObj<<<1,1>>>(d_random, size);
        generate();
    }
    void generate() {
        curandGenerateUniform(gen, d_random, size);
    }
    virtual Event* generateRandomEvent() {
        class Generate : public Action{
	    public:
	        Generate(Random *randomObj) : obj(randomObj) {}
	        void action(cudaStream_t &stream) {
	            curandGenerateUniform(obj->gen, obj->d_random, obj->size);
	        }
	        Random *obj;
	    };
        return new Event(0, SYNCHRONIZE_PRIORITY, 1, new Generate(this));
    }
    ~Random() {
        cudaFree(d_random);
        curandDestroyGenerator(gen);
    }
    float* getRandomDevice() {
        return d_random;
    }
    curandGenerator_t gen;
    float *d_random;
    uint size;
};

/*
__device__ curandState* globalState;

__device__ float cuRandom(uint id) {
    curandState localState = globalState[id % RANDOM_ELEMENTS];
    float rndval = curand_uniform( &localState );
    globalState[id % RANDOM_ELEMENTS] = localState;
    return rndval;
}

template<class T>
__device__ T cuRand(uint id, T minValue, T maxValue) {
    float rndval = cuRandom(id);
    rndval = rndval * (maxValue - minValue) + minValue;
    return truncf(rndval);
}

template<class T>
__device__ T cuRand(uint id, T maxValue) {
    return cuRand(id, (T)0, maxValue);
}

__global__ void registerRandState(curandState *state, uint elements) {
    globalState = state;
    RANDOM_ELEMENTS = elements;
}

__global__ void initRandomKernel(curandState *state, size_t size, long seed) {
    uint idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < size)
        curand_init ( seed, idx, 0, &state[idx] );
}

class Random {
public:
    Random(long seed, uint elements) : m_seed(seed), m_elements(elements){
		cudaMalloc(&state, sizeof(curandState)*elements);
		CHECK_ERROR
		registerRandState<<<1,1>>>(state, elements);
	}
	virtual Event* generateRandomEvent() {
	    class Generate : public Action{
	    public:
	        Generate(Random *randomObj) : obj(randomObj) {}
	        void action(cudaStream_t &stream) {
	            uint blocks = BLOCKS(obj->m_elements);
		        initRandomKernel<<<blocks, THREADS, 0, stream>>>(obj->state, obj->m_elements, obj->m_seed);
		        CHECK_ERROR
	        }
	        Random *obj;
	    };
        return new Event(0, COLLECTION_PRIORITY-1, 0, new Generate(this));
    }
    ~Random() {
		cudaFree(state);
	}
private:
    curandState *state;
    long m_seed;
    uint m_elements;
};
*/
#endif

