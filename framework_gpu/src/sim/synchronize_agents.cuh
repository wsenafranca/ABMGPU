#ifndef SYNCHRONIZEAGENTS_CUH
#define SYNCHRONIZEAGENTS_CUH

template<class A>
__global__ void countAgentsKernel(A *agents, uint quantity, uint *quantities, uint *numDeaths, uint *numRebirths) {
    uint idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < quantity) {
        A *ag = &agents[idx];
        
        if(ag->toDie) {
            ag->dead = true;
            ag->toDie = false;
            Cell *cell = ag->cell;
            atomicSub(&(quantities[cell->cid]), 1);
        }
        if(ag->dead) {
            atomicAdd(numDeaths, 1);
        }
        else if(ag->children > 0) {
            atomicAdd(numRebirths, ag->children);
        }
    }
}

template<class A>
__global__ void findSlots(A *agents, uint quantity, uint *slots, uint slotSize, uint *parents, uint parentSize, uint *positions) {
    uint idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < quantity) {
        A *ag = &agents[idx];
        if(ag->dead) {
            uint p = atomicAdd(&(positions[0]), 1);
            if(p < slotSize)
                slots[p] = idx;
        }
        else {
            uint children = ag->children;
            for(uint i = 0; i < children; i++) {
                uint p = atomicAdd(&(positions[1]), 1);
                if(p < parentSize)
                    parents[p] = idx;
            }
            ag->children = 0;
        }
    }
}

template<class A>
__global__ void rebirthAgentsKernel(A *agents, uint quantity, uint *slots, uint slotSize, uint *parents, uint numRebirths) {
    uint idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < numRebirths) {        
        A *parent = &agents[parents[idx]];
        uint id = idx < slotSize ? slots[idx] : quantity+idx-slotSize;
        
        A ag;
        ag.id = id;
        ag.init();
        ag.move(parent->cell);
        ag.clone(parent);
        agents[id] = ag;
    }
}

template<class A>
__global__ void syncMoveKernel(A *agents, uint quantity, uint *quantities) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < quantity) {
        A *ag = &agents[i];
        if(!ag->dead && ag->nextCell && ag->nextCell != ag->cell) {
            Cell *newCell = ag->nextCell;
            Cell *oldCell = ag->cell;
            if(oldCell) atomicSub(&(quantities[oldCell->cid]), 1);
            atomicAdd(&(quantities[newCell->cid]), 1);
            ag->cell = newCell;
        }
        ag->nextCell = NULL;
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
    
    countAgentsKernel<<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size, cs->getQuantitiesDevice(), 
                                                                soc->getNumDeathsDevice(), soc->getNumRebirthsDevice());
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
        rebirthAgentsKernel<<<rblocks, THREADS>>>(soc->getAgentsDevice(), quantity, d_slots, slotSize, d_parents, numRebirths);
        CHECK_ERROR;
        
        cudaFree(d_parents);
        cudaFree(d_positions);
        if(slotSize > 0) cudaFree(d_slots);
        
        soc->size = quantity+numRebirths-slotSize;
    }
    
    blocks = BLOCKS(soc->size);
    syncMoveKernel<<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size, cs->getQuantitiesDevice());
    CHECK_ERROR
}

#endif

