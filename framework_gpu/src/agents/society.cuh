#ifndef SOCIETY_CUH
#define SOCIETY_CUH

#include "../utils/utils.cuh"
#include "agent.cuh"

template<class A>
__global__ void initializeSocietyKernel(A *agents, uint size) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        A ag;
        ag.id = i;
        ag.init();
        agents[i] = ag;
    }
}

template<class A>
class Society{
public:
    Society(uint size, uint capacity) {
        this->size = size;
        this->capacity = capacity;
        cudaMalloc(&agents, sizeof(A)*capacity);
        CHECK_ERROR
        cudaMalloc(&numDeaths, sizeof(uint));
        CHECK_ERROR
        cudaMalloc(&numRebirths, sizeof(uint));
        CHECK_ERROR
    }
    
    ~Society() {
        cudaFree(agents);
        CHECK_ERROR
        cudaFree(numDeaths);
        CHECK_ERROR
        cudaFree(numRebirths);
        CHECK_ERROR
    }
    
    void init() {
        uint blocks = BLOCKS(size);
        initializeSocietyKernel<<<blocks, THREADS>>>(agents, size);
        CHECK_ERROR
    }
    
    A *agents;
    uint size, capacity;
    uint *numDeaths, *numRebirths;
};

#endif
