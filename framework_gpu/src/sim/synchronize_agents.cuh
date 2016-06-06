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

template<class A, class C>
void syncAgents(Society<A> *soc, CellularSpace<C> *cs) {
    uint blocks;
    blocks = BLOCKS(soc->size);
    
    uint capacity = soc->capacity;
    uint quantity = soc->size;
    
    cudaMemset(soc->getNumDeathsDevice(), 0, sizeof(uint)*2);
    CHECK_ERROR;
    
    countAgentsKernel<<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size, soc->getNumDeathsDevice(), soc->getNumRebirthsDevice());
    CHECK_ERROR;
    uint numDeaths, numRebirths;
    cudaMemcpy(&numRebirths, soc->getNumRebirthsDevice(), sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numDeaths, soc->getNumDeathsDevice(), sizeof(uint), cudaMemcpyDeviceToHost);
    
    uint slotSize = min(numDeaths, numRebirths);
    numRebirths = (quantity+numRebirths-slotSize >= capacity) ? capacity+slotSize-quantity : numRebirths;
    if(numRebirths > 0) {
        uint *d_parents, *d_slots=NULL;
        uint *d_positions;
        cudaMalloc(&d_parents, sizeof(uint)*numRebirths);
        cudaMalloc(&d_positions, sizeof(uint)*2);
        cudaMemset(d_positions, 0, sizeof(uint)*2);
        
        if(slotSize > 0) cudaMalloc(&d_slots, sizeof(uint)*slotSize);
        findSlots<<<blocks, THREADS>>>(soc->getAgentsDevice(), quantity,
                                       d_slots, slotSize, 
                                       d_parents, numRebirths, d_positions);
        CHECK_ERROR;
        
        uint rblocks = BLOCKS(numRebirths);
        rebirthAgentsKernel<<<rblocks, THREADS>>>(soc->getAgentsDevice(), soc->getPastDevice(), quantity, 
                                                  d_slots, slotSize, d_parents, numRebirths);
        CHECK_ERROR;
        
        cudaFree(d_parents);
        cudaFree(d_positions);
        if(slotSize > 0) cudaFree(d_slots);
        
        soc->size = quantity+numRebirths-slotSize;
    }
    
    // past = present
    soc->synchronize();
}

/*
template<class A>
__global__ void compress(A *agentsIn, A *agentsOut, uint *indices, uint size) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        agentsOut[i] = agentsIn[indices[i]];
        indices[i] = i;
    }
}

template<class A>
__global__ void reproduce(A *agents, uint *indices, uint size, uint *pos, uint capacity) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        A *parent = &agents[i];
        uint children = parent->children;
        if(children > 0) {
            uint p = atomicAdd(pos, children) + size;
            uint id;
            for(uint j = 0; j < children; j++) {
                id = p+j;
                if(id < capacity) {
                    A ag;
                    ag.init();
                    ag.cell = parent->cell;
                    ag.clone(parent);
                    agents[id] = ag;
                    indices[id] = id;
                }
            }
        }
    }
}

template<class A>
struct isAlive{
    isAlive(A *agents) : d_agents(agents) {}
    A *d_agents;
    __device__ bool operator()(const uint i) {
        return !d_agents[i].isDead();
    }
    __device__ bool operator()(const A &a) {
        return !a.isDead();
    }
    __device__ bool operator()(const thrust::tuple<uint, A> &t) {
        return !d_agents[thrust::get<0>(t)].isDead();
    }
};

template<class A, class C>
void syncAgents(Society<A> *soc, CellularSpace<C> *cs) {    
    uint newSize = thrust::partition(thrust::device, soc->getIndicesDevice(), soc->getIndicesDevice() + soc->size, 
                   isAlive<A>(soc->getAgentsDevice())) - soc->getIndicesDevice();
    CHECK_ERROR
    //printf("%u\n", newSize);
    uint blocks = blocks = BLOCKS(newSize);
    
    if(newSize != soc->size) {
        blocks = BLOCKS(soc->size);
        //compress<<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->getPastDevice(), soc->getIndicesDevice(), soc->size);
        //cudaMemcpy(soc->getAgentsDevice(), soc->getPastDevice(), sizeof(A)*soc->size, cudaMemcpyDeviceToDevice);
    }
    
    cudaMemset(soc->getNumRebirthsDevice(), 0, sizeof(uint));
    reproduce<<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->getIndicesDevice(), soc->size, soc->getNumRebirthsDevice(), soc->capacity);
    uint numRebirths;
    cudaMemcpy(&numRebirths, soc->getNumRebirthsDevice(), sizeof(uint), cudaMemcpyDeviceToHost);
    soc->size = min(soc->size+numRebirths, soc->capacity);
    soc->synchronize();
}
*/

#endif


