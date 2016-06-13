#ifndef SOCIETY_CUH
#define SOCIETY_CUH

#include "../utils/utils.cuh"
#include "agent.cuh"

template<class A>
__global__ void initializeSocietyKernel(A *agents, const uint size) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        A ag;
        ag.init();
        agents[i] = ag;
    }
}

template<class A>
class Society : public Collection{
public:
    Society(uint size, uint capacity) : Collection(size, capacity) {
        uint totalMem = 0;
        agentsIndex = totalMem;
        totalMem += sizeof(A)*capacity;     // alloc agents
        
        pastIndex = totalMem;
        totalMem += sizeof(A)*capacity;     // alloc past
        
        numDeathsIndex = totalMem;
        totalMem += sizeof(uint);          // alloc numDeaths
        
        numRebirthsIndex = totalMem;
        totalMem += sizeof(uint);          // alloc numRebirths
        
        alloc(totalMem);
    }
    
    ~Society() {
    }
    
    Event* initializeEvent() {
        class Init : public Action{
        public:
            Init(Society<A> *society) : soc(society) {}
            void action(cudaStream_t &stream) {
                uint blocks = BLOCKS(soc->size);
                initializeSocietyKernel<<<blocks, THREADS, 0, stream>>>(soc->getAgentsDevice(), soc->size);
                CHECK_ERROR
            }
            Society<A> *soc;
        };
        return new Event(0, COLLECTION_PRIORITY, 0, new Init(this));
    }
    
    A* getAgentsDevice() {
        return (A*)(getMemPtrDevice()+getIndexOnMemoryDevice()+agentsIndex);
    }
    
    A* getPastDevice() {
        return (A*)(getMemPtrDevice()+getIndexOnMemoryDevice()+pastIndex);
    }
     
    uint* getNumDeathsDevice() {
        return (uint*)(getMemPtrDevice()+getIndexOnMemoryDevice()+numDeathsIndex);
    }
    
    uint* getNumRebirthsDevice() {
        return (uint*)(getMemPtrDevice()+getIndexOnMemoryDevice()+numRebirthsIndex);
    }
    
private:
    uint agentsIndex;
    uint pastIndex;
    uint indicesIndex;
    uint numDeathsIndex;
    uint numRebirthsIndex;
};

#endif

