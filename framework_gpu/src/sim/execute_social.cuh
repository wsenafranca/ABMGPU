#ifndef EXECUTESOCIAL_CUH
#define EXECUTESOCIAL_CUH

template<SocialChange change, class A>
__global__ void executeKernel(A *agents, uint quantity) {
    uint i = threadIdx.x+blockIdx.x*blockDim.x;
    if(i < quantity && !agents[i].isDead()) {
        change(&agents[i]);
    }
}

template<SocialChange change, class A>
void execute(Society<A> *soc) {
	if(soc->size == 0) return;
	
    uint blocks = BLOCKS(soc->size);

    executeKernel<change, A><<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size);
    CHECK_ERROR;
}

template<SocialChangePair change, class A>
__global__ void executeKernel(A *agents,  A *past, uint size, uint *neighborhood) {
    uint idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size) {        
        A *ag = &agents[idx];
        if(ag->isDead()) return;
        
        Cell *cell = ag->cell;
        uint2 pos = cell->getPos();
        int x = pos.x;
        int y = pos.y;
        
        uint init = tex2D(beginsRef, x, y);
        uint end = tex2D(endsRef, x, y);
        
        for(uint j = init; j < end; j++) {
            A *ag2 = &agents[neighborhood[j]];
            if(ag != ag2 && !ag2->isDead()) {
                change(ag, ag2);
            }
        }
    }
}

template<SocialChangePair change, class A>
__global__ void executeKernel(A *agents,  A *past, uint size, uint *neighborhood, const int n, const int m, 
                              const int nxdim, const int nydim) {
    uint idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size) {        
        A *ag = &agents[idx];
        if(ag->isDead()) return;
        Cell *cell = ag->cell;
        uint2 pos = cell->getPos();
        int x = truncf(pos.x/m);
        int y = truncf(pos.y/n);
        int x1 = pos.x;
        int y1 = pos.y;
        
        uint* ends[9];
        uint* inits[9];
        uint neighs = 0;
        for(int ny = y-1; ny <= y+1; ny++) {
            for(int nx = x-1; nx <= x+1; nx++) {
                if(nx >= 0 && nx < nxdim && ny >= 0 && ny < nydim) {
                    uint begin = tex2D(beginsRef, nx, ny);
                    uint end = tex2D(endsRef, nx, ny);
                    if(end > 0) {
                        inits[neighs] = &neighborhood[begin];
                        ends[neighs] = &neighborhood[end];
                        neighs++;
                    }
                }
            }
        }
        uint *p;
        for(uint i = 0; i < neighs; i++) {
            for(p = inits[i]; p < ends[i]; p++) {
                A *ag2 = &past[*p];
                if(ag != ag2 && !ag2->isDead() && MANHATTAN_DIST(x1, y1, ag2->cell->getX(), ag2->cell->getY()) <= n) {
                    change(ag, ag2);
                }
            }
        }
    }
}

template<SocialChangePair change, class A, class C>
void execute(Society<A> *soc, CellularSpace<C> *cs, Neighborhood<A, C> *nb) {
	if(soc->size == 0) return;
	
    uint blocks = BLOCKS(soc->size);
    
    if(nb->n == 0 && nb->m == 0) {
        executeKernel<change><<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->getPastDevice(), soc->size, 
                                                   nb->getNeighborhoodDevice());
    }
    else {
        executeKernel<change><<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->getPastDevice(), soc->size, 
                                                   nb->getNeighborhoodDevice(), nb->n, nb->m, 
                                                   nb->neighborhoodXDim, nb->neighborhoodYDim);
    }
    CHECK_ERROR;
}

#endif


