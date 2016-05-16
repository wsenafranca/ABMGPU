#ifndef PLACEMENTS
#define PLACEMENTS

template<class A, class C>
__global__ void placementKernel(A *agents, uint size, C *cells, uint *quantities, uint xdim, uint ydim) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        uint cid = cuRand(i, ydim*xdim) % (ydim*xdim);
        A *ag = &agents[i];
        C *cell = &(cells[cid]);
        atomicAdd(&(quantities[cid]), 1);
        ag->cell = cell;
        ag->nextCell = NULL;        
    }
}

template<class A, class C>
void placement(Society<A> *soc, CellularSpace<C> *cs) {
    uint blocks = BLOCKS(soc->size);
    placementKernel<<<blocks, THREADS>>>(soc->agents, soc->size, cs->cells, cs->quantities, cs->xdim, cs->ydim);
    CHECK_ERROR
}

#endif
