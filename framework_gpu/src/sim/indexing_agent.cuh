#ifndef INDEXINGAGENT_CUH
#define INDEXINGAGENT_CUH

#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

template<class A>
__global__ void findInhabitedKernel(A *agents, uint size, uint *inhabited, uint n, uint m, uint neighXDim) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        A *ag = &agents[i];
        
        Cell *cell = ag->cell;
        int x = m > 0 ? truncf(cell->getX()/m) : cell->getX();
        int y = n > 0 ? truncf(cell->getY()/n) : cell->getY();
        uint ncid = y*neighXDim + x;

        inhabited[i] = ncid;
    }
}

template<class A>
__global__ void sortAgentsKernel(A *agents, uint size, uint *neighborhood, uint *pos, uint n, uint m, uint neighborhoodXDim) {
    uint idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size) {
        A *ag = &agents[idx];
        Cell *cell = ag->cell;
        uint x = m > 0 ? truncf(cell->getX()/m) : cell->getX();
        uint y = n > 0 ? truncf(cell->getY()/n) : cell->getY();
        uint ncid = y*neighborhoodXDim + x;
        uint init = tex2D(beginsRef, x, y);
        uint p = atomicAdd(&(pos[ncid]), 1);
        neighborhood[init+p] = idx;
    }
}

__global__ void indexing(uint *cellids, uint size, uint *begins, uint *ends, uint neighXDim) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        if(i == 0 || cellids[i] != cellids[i-1]) 
            begins[cellids[i]] = i;
        if(i == size-1 || cellids[i] != cellids[i+1])
            ends[cellids[i]] = i+1;
    }
}

template<class A, class C>
class IndexingAgentAction : public Action{
public:
    IndexingAgentAction(Society<A> *society, CellularSpace<C> *cellSpace, Neighborhood<A,C> *neighborhood) 
        : soc(society), cs(cellSpace), nb(neighborhood){}
        
    void action(cudaStream_t &stream) {
        nb->clear();
        uint blocks;
        blocks = BLOCKS(soc->size);
        
        findInhabitedKernel<<<blocks, THREADS, 0, stream>>>(soc->getAgentsDevice(), soc->size, 
                                                            nb->getInhabitedDevice(), nb->n, nb->m, nb->neighborhoodXDim);
        thrust::sort(thrust::cuda::par.on(stream), nb->getInhabitedDevice(), nb->getInhabitedDevice()+soc->size);
        
        blocks = BLOCKS(soc->size);
        indexing<<<blocks, THREADS, 0, stream>>>(nb->getInhabitedDevice(), soc->size, nb->getOffsetDevice(), 
                                                 nb->getQuantitiesDevice(), nb->neighborhoodXDim);
        nb->syncTexture(stream);
    
	    blocks = BLOCKS(soc->size);
	    sortAgentsKernel<<<blocks, THREADS, 0, stream>>>(soc->getAgentsDevice(), soc->size, 
	                                                     nb->getNeighborhoodDevice(), nb->getPosDevice(), 
	                                                     nb->n, nb->m, nb->neighborhoodXDim);
    }
    Society<A> *soc;
    CellularSpace<C> *cs;
    Neighborhood<A,C> *nb;
};

#endif


