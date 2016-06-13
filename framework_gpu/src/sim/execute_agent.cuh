#ifndef EXECUTEAGENT_CUH
#define EXECUTEAGENT_CUH

template<AgentExecute exec, class A>
__global__ void executeKernel(A *agents, uint size) {
    uint i = threadIdx.x+blockIdx.x*blockDim.x;
    if(i < size && !agents[i].isDead()) {
        exec(&agents[i]);
    }
}

template<AgentExecute exec, class A>
class AgentExecuteAction : public Action{
public:
    AgentExecuteAction(Society<A> *soc) : p_soc(soc) {}
    void action(cudaStream_t &stream) {
        if(p_soc->size > 0) {
            uint blocks = BLOCKS(p_soc->size);
            executeKernel<exec><<<blocks, THREADS, 0, stream>>>(p_soc->getAgentsDevice(), p_soc->size);
        }
    }
    Society<A> *p_soc;
};

template<AgentSpatiallyExecute exec, class A, class C>
__global__ void executeKernel(A *agents, uint size, Cell *cells, uint xdim, uint ydim) {
    uint i = threadIdx.x+blockIdx.x*blockDim.x;
    if(i < size && !agents[i].isDead()) {
        exec(&agents[i], cells, xdim, ydim);
    }
}

template<AgentSpatiallyExecute exec, class A, class C>
class AgentSpatiallyExecuteAction : public Action{
public:
    AgentSpatiallyExecuteAction(Society<A> *soc, CellularSpace<C> *cs) : p_soc(soc), p_cs(cs) {}
    void action(cudaStream_t &stream) {
        if(p_soc->size > 0) {
            uint blocks = BLOCKS(p_soc->size);
            executeKernel<exec, A><<<blocks, THREADS, 0, stream>>>(p_soc->getAgentsDevice(), p_soc->size, 
                                                                   p_cs->getCellsDevice(), p_cs->xdim, p_cs->ydim);
        }
    }
    Society<A> *p_soc;
    CellularSpace<C> *p_cs;
};

template<AgentSpatiallyNeighborhoodExecute exec, class A, class C>
__global__ void executeKernel(A *agents, A *past, uint size, C *cells, uint xdim, uint ydim, 
                              uint *neighborhood, uint n, uint m, uint nxdim, uint nydim) {
    uint idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size && !agents[idx].isDead()) {
        A *ag = &agents[idx];
        Iterator<uint> begins[9];
        Iterator<uint> ends[9];
        uint neighs;
        NeighborhoodIterator::create(ag, neighborhood, n, m, xdim, nydim, begins, ends, &neighs);
        
        NeighborhoodIterator begin(begins[0], begins, ends);
        NeighborhoodIterator end(begins[neighs], begins+neighs, ends);
        
        exec(ag, past, cells, xdim, ydim, begin, end);
    }
}

template<AgentSpatiallyNeighborhoodExecute exec, class A, class C>
class AgentSpatiallyNeighborhoodAction : public Action {
public:
    AgentSpatiallyNeighborhoodAction(Society<A> *soc, CellularSpace<C> *cs, Neighborhood<A, C> *nb, uint init, uint partSize) 
        : p_soc(soc), p_cs(cs), p_nb(nb), ini(init), psize(partSize){}
        
    void action(cudaStream_t &stream) {
        if(p_soc->size > 0) {
            uint blocks = BLOCKS(psize);
            executeKernel<exec, A><<<blocks, THREADS, 0, stream>>>(p_soc->getAgentsDevice()+ini, p_soc->getPastDevice(), psize, 
                                                                   p_cs->getCellsDevice(), p_cs->xdim, p_cs->ydim,
                                                                   p_nb->getNeighborhoodDevice(), p_nb->n, p_nb->m,
                                                                   p_nb->neighborhoodXDim, p_nb->neighborhoodYDim);
        }
    }
    
    Society<A> *p_soc;
    CellularSpace<C> *p_cs;
    Neighborhood<A, C> *p_nb;
    uint ini, psize;
};

namespace ExecuteEvent{
    template<AgentExecute exec, class A>
    static Event* execute(Society<A> *soc, uint time, uint priority, uint period) {
        return new Event(time, priority, period, new AgentExecuteAction<exec, A>(soc));
    }
    
    template<AgentSpatiallyExecute exec, class A, class C>
    static Event* execute(Society<A> *soc, CellularSpace<C> *cs, uint time, uint priority, uint period) {
        return new Event(time, priority, period, 
                         new AgentSpatiallyExecuteAction<exec, A, C>(soc, cs));
    }
    
    template<AgentSpatiallyNeighborhoodExecute exec, class A, class C>
    static Event* execute(Society<A> *soc, CellularSpace<C> *cs, Neighborhood<A, C> *nb, uint time, uint priority, uint period, uint init, uint partSize) {
        return new Event(time, priority, period, new AgentSpatiallyNeighborhoodAction<exec, A, C>(soc, cs, nb, init, partSize));
    }
};

#endif


