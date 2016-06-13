#ifndef SYNCHRONIZEAGENTS_CUH
#define SYNCHRONIZEAGENTS_CUH

#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

template<class A>
__global__ void countAgentsKernel(A *agents, const uint quantity, uint *numDeaths, uint *numRebirths) {
    uint idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < quantity) {
        A *ag = &agents[idx];
        if(ag->isDead()) {
            atomicAdd(numDeaths, 1);
        }
        else if(ag->children > 0) {
            atomicAdd(numRebirths, ag->children);
        }
    }
}

template<class A>
__global__ void findSlots(A *agents, const uint quantity, uint *slots, const uint slotSize, uint *parents, const uint parentSize, uint *positions) {
    uint idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < quantity) {
        A *ag = &agents[idx];
        if(ag->isDead()) {
            uint p = atomicAdd(&(positions[0]), 1);
            if(p < slotSize)
                slots[p] = idx;
        }
        else {
            uint children = ag->children;
            if(children > 0) {
                uint p = atomicAdd(&(positions[1]), children);
                for(uint i = 0; i < children; i++) {
                    if(p+i < parentSize)
                        parents[p+i] = idx;
                }
                ag->children = 0;
            }
        }
    }
}

template<class A>
__global__ void rebirthAgentsKernel(A *agents, A *past, const uint quantity, const uint *slots, 
                                    const uint slotSize, const uint *parents, const uint numRebirths) {
    uint idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < numRebirths) {        
        A *parent = &agents[parents[idx]];
        uint id = idx < slotSize ? slots[idx] : quantity+idx-slotSize;
        
        A ag;
        ag.init();
        ag.move(parent->cell);
        ag.clone(parent);
        agents[id] = ag;
    }
}

template<class A>
class SynchronizeAgentAction : public Action{
public:
    SynchronizeAgentAction(Society<A> *society) : soc(society) {}
    void action(cudaStream_t &stream) {
        uint blocks;
        blocks = BLOCKS(soc->size);
        
        uint capacity = soc->capacity;
        uint quantity = soc->size;
        
        cudaMemset(soc->getNumDeathsDevice(), 0, sizeof(uint)*2);
        
        countAgentsKernel<<<blocks, THREADS, 0, stream>>>(soc->getAgentsDevice(), soc->size, 
                                                          soc->getNumDeathsDevice(), soc->getNumRebirthsDevice());
        uint numDeaths, numRebirths;
        cudaMemcpyAsync(&numRebirths, soc->getNumRebirthsDevice(), sizeof(uint), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&numDeaths, soc->getNumDeathsDevice(), sizeof(uint), cudaMemcpyDeviceToHost, stream);
        
        uint slotSize = min(numDeaths, numRebirths);
        numRebirths = (quantity+numRebirths-slotSize >= capacity) ? capacity+slotSize-quantity : numRebirths;
        if(numRebirths > 0) {
            uint *d_parents, *d_slots=NULL;
            uint *d_positions;
            cudaMalloc(&d_parents, sizeof(uint)*numRebirths);
            cudaMalloc(&d_positions, sizeof(uint)*2);
            cudaMemset(d_positions, 0, sizeof(uint)*2);
            
            if(slotSize > 0) cudaMalloc(&d_slots, sizeof(uint)*slotSize);
            findSlots<<<blocks, THREADS, 0, stream>>>(soc->getAgentsDevice(), quantity,
                                                      d_slots, slotSize, 
                                                      d_parents, numRebirths, d_positions);
            
            uint rblocks = BLOCKS(numRebirths);
            rebirthAgentsKernel<<<rblocks, THREADS, 0, stream>>>(soc->getAgentsDevice(), soc->getPastDevice(), quantity, 
                                                                 d_slots, slotSize, d_parents, numRebirths);
            
            cudaFree(d_parents);
            cudaFree(d_positions);
            if(slotSize > 0) cudaFree(d_slots);
            
            soc->size = quantity+numRebirths-slotSize;
        }
        cudaMemcpyAsync(soc->getPastDevice(), soc->getAgentsDevice(), sizeof(A)*soc->size, cudaMemcpyDeviceToDevice, stream);
    }
    
    Society<A> *soc;
};

#endif


