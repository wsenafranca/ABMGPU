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
        uint totalMem = 0;
        totalMem += sizeof(A)*capacity;     // alloc agents
        totalMem += sizeof(uint);           // alloc numDeaths
        totalMem += sizeof(uint);           // alloc numRebirths
        
        agentsIndex = 0;
        numDeathsIndex = sizeof(A)*capacity;
        numRebirthsIndex = sizeof(A)*capacity+sizeof(uint);
        printf("%u\n", totalMem);
        index = Environment::getEnvironment()->alloc(totalMem);
    }
    
    ~Society() {
    }
    
    void init(cudaStream_t *stream) {
        uint blocks = BLOCKS(size);
        initializeSocietyKernel<A><<<blocks, THREADS, 0, *stream>>>(getAgentsDevice(), size);
        CHECK_ERROR
    }
    
    A* getAgentsDevice() {
        return (A*)(Environment::getEnvironment()->getGlobalMemory()+index+agentsIndex);
    }
    
    uint* getNumDeathsDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+numDeathsIndex);
    }
    
    uint* getNumRebirthsDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+numRebirthsIndex);
    }
    
    uint size, capacity;
    
private:
    uint index;
    uint agentsIndex;
    uint numDeathsIndex;
    uint numRebirthsIndex;
};

#endif
