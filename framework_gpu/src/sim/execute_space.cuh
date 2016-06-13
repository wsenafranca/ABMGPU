#ifndef EXECUTESPACE_CUH
#define EXECUTESPACE_CUH

template<CellChange change, class C>
__global__ void executeKernel(C *cells, uint size) {
    uint i = threadIdx.x+blockIdx.x*blockDim.x;
    if(i < size) {
        change(&cells[i]);
    }
}

template<CellChange exec, class C>
class CellExecuteAction : public Action{
public:
    CellExecuteAction(CellularSpace<C> *cs) : p_cs(cs) {}
    void action(cudaStream_t &stream) {
        uint blocks = BLOCKS(p_cs->size);
        executeKernel<exec><<<blocks, THREADS, 0, stream>>>(p_cs->getCellsDevice(), p_cs->size);
    }
    CellularSpace<C> *p_cs;
};

template<CellSpatiallyChange exec, class C>
__global__ void executeKernel(C *cells, C *past, uint size, uint xdim, uint ydim) {
    uint i = threadIdx.x+blockIdx.x*blockDim.x;
    if(i < size) {
        exec(&cells[i], past, xdim, ydim);
    }
}

template<CellSpatiallyChange exec, class C>
class CellSpatiallyExecuteAction : public Action{
public:
    CellSpatiallyExecuteAction(CellularSpace<C> *cs) 
        : p_cs(cs) {}
    void action(cudaStream_t &stream) {
        uint blocks = BLOCKS(p_cs->size);
        executeKernel<exec><<<blocks, THREADS, 0, stream>>>(p_cs->getCellsDevice(), p_cs->getPastDevice(), 
                                                            p_cs->size, p_cs->xdim, p_cs->ydim);
    }
    CellularSpace<C> *p_cs;
};

namespace ExecuteEvent{
    template<CellChange exec, class C>
    static Event* execute(CellularSpace<C> *cs, uint time, uint priority, uint period) {
        return new Event(time, priority, period, new CellExecuteAction<exec, C>(cs));
    }
    
    template<CellSpatiallyChange exec, class C>
    static Event* execute(CellularSpace<C> *cs, uint time, uint priority, uint period) {
        return new Event(time, priority, period, new CellSpatiallyExecuteAction<exec, C>(cs));
    }
};

#endif


