#ifndef EXECUTESPATIAL_CUH
#define EXECUTESPATIAL_CUH

template<SpaceSocialChange change, class A, class C>
__global__ void executeKernel(A *agents, uint quantity, C *cs, uint xdim, uint ydim) {
    uint i = threadIdx.x+blockIdx.x*blockDim.x;
    if(i < quantity) {
        A *ag = &agents[i];
        if(!ag->dead) change(ag, cs, xdim, ydim);
    }
}

template<SpaceSocialChange change, class A, class C>
void execute(Society<A> *soc, CellularSpace<C> *cs) {
	if(soc->size == 0) return;
	
    uint blocks = BLOCKS(soc->size);
    executeKernel<change, A><<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size,
                                                  cs->getCellsDevice(), cs->xdim, cs->ydim);
    CHECK_ERROR;
}



template<SpaceSocialChangePair change, class A, class C>
__device__ void searchNeighs(A *ag, C *cell, A **neighborhood, uint *offset, C *cells, uint *quantities, uint xdim, uint ydim) {
    uint q = quantities[cell->cid];
    for(uint j = 0; j < q; j++) {
        A *ag2 = neighborhood[offset[cell->cid]+j];
        if(ag != ag2 && !ag->dead) {
            change(ag, ag2, cells, xdim, ydim);
        }
    }
}

template<SpaceSocialChangePair change, class A, class C>
__global__ void executeKernel(A *agents, uint size, C *cells, uint *quantities, uint xdim, uint ydim, 
                                         A **neighborhood, uint *offset, int n, int m) {
    uint idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size) {        
        A *ag = &agents[idx];
        if(ag->dead) return;
        
        C *cell = (C*)ag->cell;
        int x = cell->getX();
        int y = cell->getX();
        for(int i = -n; i <= n; i++) {
            for(int j = -m; j <= m; j++) {
                int nx = x+j;
                int ny = y+i;
                if(nx >= 0 && nx < (int)xdim && ny >= 0 && ny < (int)ydim){
                    uint newcid = ny*xdim + nx;
                    searchNeighs<change>(ag, &(cells[newcid]), neighborhood, offset, cells, quantities, xdim, ydim);
                }
            }
        }
    }
}

template<SpaceSocialChangePair change, class A, class C>
void execute(Society<A> *soc, CellularSpace<C> *cs, Neighborhood<A, C> *nb) {
	if(soc->size == 0) return;
	
    uint blocks = BLOCKS(soc->size);
    executeKernel<change, A><<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size, 
                                               cs->getCellsDevice(), cs->getQuantitiesDevice(), cs->xdim, cs->ydim, 
                                               nb->getNeighborhoodDevice(), nb->getOffsetDevice(), nb->n, nb->m);
    CHECK_ERROR;
}

template<SpatialChange schange, class C>
__global__ void changeKernel(C *cs, uint len) {
    uint i = threadIdx.x+blockIdx.x*blockDim.x;
    if(i < len) {
        C *c = &cs[i];
        schange(c);
    }
}

template<SpatialChange schange, class C>
void change(CellularSpace<C> *cs) {
    uint blocks = BLOCKS(cs->xdim*cs->ydim);
    changeKernel<schange><<<blocks, THREADS>>>(cs->getCellsDevice(), cs->xdim*cs->ydim);
    CHECK_ERROR;
}

#endif

