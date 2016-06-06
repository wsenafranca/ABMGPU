#ifndef PLACEMENTS
#define PLACEMENTS

template<class A, class C>
__global__ void placementKernel(A *agents, const uint size, C *cells, const uint xdim, const uint ydim) {
    const uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        uint cid = cuRand(i, ydim*xdim) % (ydim*xdim);
        A *ag = &agents[i];
        ag->move(&cells[cid]);
    }
}

template<class A, class C>
void placement(Society<A> *soc, CellularSpace<C> *cs) {
    uint blocks = BLOCKS(soc->size);
    
    placementKernel<A><<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size, 
                                            cs->getCellsDevice(), cs->xdim, cs->ydim);
    CHECK_ERROR
}

#endif

