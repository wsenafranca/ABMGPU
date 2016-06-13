#ifndef CELLULARSPACE_CUH
#define CELLULARSPACE_CUH

#include "../utils/utils.cuh"
#include "cell.cuh"

template<class C>
__global__ void initializeCellularSpaceKernel(C *cells, const uint xdim, const uint ydim, const uint len) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < len) {
        C c;
        c.cid = i;
        c.pos.x = i%xdim;
        c.pos.y = i/xdim;
        
        cells[i] = c;
    }
}

template<class C>
class CellularSpace : public Collection{
public:
    CellularSpace(uint xDim, uint yDim, bool past = true) : Collection(xDim*yDim, xDim*yDim), xdim(xDim), ydim(yDim) {
        cellsIndex = alloc(sizeof(C)*xdim*ydim);
        pastIndex = past ? alloc(sizeof(C)*xdim*ydim) : cellsIndex;
    }
    
    ~CellularSpace() {}
    
    Event* initializeEvent() {
        class Init : public Action{
        public:
            Init(CellularSpace<C> *cellularspace) : cs(cellularspace) {}
            void action(cudaStream_t &stream) {
                uint blocks = BLOCKS(cs->size);
                initializeCellularSpaceKernel<<<blocks, THREADS, 0, stream>>>(cs->getCellsDevice(), cs->xdim, cs->ydim, cs->size);
                CHECK_ERROR
            }
            CellularSpace<C> *cs;
        };
        return new Event(0, COLLECTION_PRIORITY, 0, new Init(this));
    }
    
    C* getCellsDevice() {
        return (C*)(getMemPtrDevice() + getIndexOnMemoryDevice() + cellsIndex);
    }
    
    C* getPastDevice() {
        return (C*)(getMemPtrDevice() + getIndexOnMemoryDevice() + pastIndex);
    }
    
    const uint xdim, ydim;
    uint cellsIndex;
    uint pastIndex;
};

#endif

