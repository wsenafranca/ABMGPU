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
class CellularSpace{
public:
    CellularSpace(uint xDim, uint yDim) : xdim(xDim), ydim(yDim) {
    
        uint totalMem = 0;
        totalMem += sizeof(C)*xdim*ydim;    // alloc cells        
        cellsIndex = 0;
        
        index = Environment::getEnvironment()->alloc(totalMem);
    }
    
    ~CellularSpace() {
    }
    
    void init() {        
        uint blocks = BLOCKS(xdim*ydim);
        initializeCellularSpaceKernel<<<blocks, THREADS>>>(getCellsDevice(), xdim, ydim, xdim*ydim);
        CHECK_ERROR
    }
    
    C* getCellsDevice() {
        return (C*)(Environment::getEnvironment()->getGlobalMemory() + index + cellsIndex);
    }
    
    const uint xdim, ydim;
    uint cellsIndex;
    uint index;
};

#endif

