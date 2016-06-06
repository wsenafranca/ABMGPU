#ifndef SOCIETY_CUH
#define SOCIETY_CUH

#include "../utils/utils.cuh"
#include "agent.cuh"

template<class A>
__global__ void initializeSocietyKernel(A *agents, A *past, const uint size) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        A ag;
        ag.init();
        agents[i] = ag;
    }
}

template<class A>
class Society{
public:
    Society(uint Size, uint Capacity) : size(Size), capacity(Capacity) {
        uint totalMem = 0;
        agentsIndex = totalMem;
        totalMem += sizeof(A)*capacity;     // alloc agents
        
        pastIndex = totalMem;
        totalMem += sizeof(A)*capacity;     // alloc past
        
        //indicesIndex = totalMem;
        //totalMem += sizeof(uint)*capacity;
        
        numDeathsIndex = totalMem;
        totalMem += sizeof(uint);          // alloc numDeaths
        
        numRebirthsIndex = totalMem;
        totalMem += sizeof(uint);          // alloc numRebirths
        
        index = Environment::getEnvironment()->alloc(totalMem);
    }
    
    ~Society() {
    }
    
    void init() {
        uint blocks = BLOCKS(size);
        initializeSocietyKernel<<<blocks, THREADS>>>(getAgentsDevice(), getPastDevice(), size);
        cudaMemcpy(getPastDevice(), getAgentsDevice(), sizeof(A)*size, cudaMemcpyDeviceToDevice);
        CHECK_ERROR
    }
    
    void synchronize() {
        cudaMemcpy(getPastDevice(), getAgentsDevice(), sizeof(A)*size, cudaMemcpyDeviceToDevice);
    }
    
    A* getAgentsDevice() {
        return (A*)(Environment::getEnvironment()->getGlobalMemory()+index+agentsIndex);
    }
    
    A* getPastDevice() {
        return (A*)(Environment::getEnvironment()->getGlobalMemory()+index+pastIndex);
    }
    
    //uint* getIndicesDevice() {
    //    return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+indicesIndex);
    //}
     
    uint* getNumDeathsDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+numDeathsIndex);
    }
    
    uint* getNumRebirthsDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+numRebirthsIndex);
    }
    
    uint size;
    const uint capacity;
    
private:
    uint index;
    uint agentsIndex;
    uint pastIndex;
    uint indicesIndex;
    uint numDeathsIndex;
    uint numRebirthsIndex;
};

#endif

