#ifndef EXECUTESOCIAL_CUH
#define EXECUTESOCIAL_CUH

template<SocialChange change, class A>
__global__ void executeKernel(A *agents, uint quantity) {
    uint i = threadIdx.x+blockIdx.x*blockDim.x;
    if(i < quantity) {
        A *ag = &agents[i];
        if(!ag->dead) change(ag);
    }
}

template<SocialChange change, class A>
void execute(Society<A> *soc) {
	if(soc->size == 0) return;
	
    uint blocks = BLOCKS(soc->size);

    executeKernel<change, A><<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size);
    CHECK_ERROR;
}

template<SocialChangePair change, class A, class C>
__device__ void searchNeighs2(A *ag, C *cell, A **neighborhood, uint *offset, uint *quantities) {
    uint q = quantities[cell->cid];
    for(uint j = 0; j < q; j++) {
        A *ag2 = neighborhood[offset[cell->cid]+j];
        if(ag != ag2 && !ag2->dead) {
            change(ag, ag2);
        }
    }
}

template<SocialChangePair change, class A, class C>
__global__ void executeKernel2(A *agents, uint size, C *cells, uint *quantities, uint xdim, uint ydim, 
                                         A **neighborhood, uint *offset, int n, int m) {
    uint idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size) {        
        A *ag = &agents[idx];
        if(ag->dead) return;
        
        C *cell = (C*)ag->cell;
        int x = cell->getX();
        int y = cell->getY();
        for(int i = -n; i <= n; i++) {
            for(int j = -m; j <= m; j++) {
                int nx = x+j;
                int ny = y+i;
                if(nx >= 0 && nx < (int)xdim && ny >= 0 && ny < (int)ydim){
                    uint newcid = ny*xdim + nx;
                    searchNeighs2<change>(ag, &(cells[newcid]), neighborhood, offset, quantities);
                }
            }
        }
    }
}

template<SocialChangePair change, class A, class C>
void execute(Society<A> *soc, CellularSpace<C> *cs, Neighborhood<A, C> *nb) {
	if(soc->size == 0) return;
	
    uint blocks = BLOCKS(soc->size);
    
    executeKernel2<change, A><<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size, 
                                                   cs->getCellsDevice(), cs->getQuantitiesDevice(), cs->xdim, cs->ydim, 
                                                   nb->getNeighborhoodDevice(), nb->getOffsetDevice(), nb->n, nb->m);
    CHECK_ERROR;
}

#endif

