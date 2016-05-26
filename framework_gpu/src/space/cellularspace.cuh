#ifndef CELLULARSPACE_CUH
#define CELLULARSPACE_CUH

#include "../utils/utils.cuh"
#include "cell.cuh"

template<class C>
__global__ void initializeCellularSpaceKernel(C *cells, uint *quantities, 
                                                                uint *dimensions, uint len) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < len) {
        C c;
        c.cid = i;
        c.xdim = &dimensions[0];
        c.ydim = &dimensions[1];
        c.quantities = quantities;
        quantities[i] = 0;
        
        cells[i] = c;
    }
}

template<class C>
class CellularSpace{
public:
    CellularSpace(uint xdim, uint ydim) {
        this->xdim = xdim;
        this->ydim = ydim;
        
        uint totalMem = 0;
        totalMem += sizeof(C)*xdim*ydim;    // alloc cells
        totalMem += sizeof(uint)*xdim*ydim; // alloc quantities
        totalMem += sizeof(uint)*2;         // alloc dimensions
        
        cellsIndex = 0;
        quantitiesIndex = sizeof(C)*xdim*ydim;
        dimsIndex = sizeof(C)*xdim*ydim + sizeof(uint)*xdim*ydim;
        
        index = Environment::getEnvironment()->alloc(totalMem);
    }
    
    ~CellularSpace() {
    }
    
    void init(cudaStream_t *stream) {        
    	static uint dimensions[2];
    	dimensions[0] = xdim; dimensions[1] = ydim;
    	uint *dims = (uint*)(Environment::getEnvironment()->getGlobalMemory()+dimsIndex+index);
    	
    	cudaMemcpy(dims, dimensions, sizeof(uint)*2, cudaMemcpyHostToDevice);
    	CHECK_ERROR
        uint blocks = BLOCKS(xdim*ydim);
        initializeCellularSpaceKernel<<<blocks, THREADS, 0, *stream>>>(getCellsDevice(), getQuantitiesDevice(), dims, xdim*ydim);
        CHECK_ERROR
    }
    
    uint* getQuantitiesDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory() + index + quantitiesIndex);
    }
    
    C* getCellsDevice() {
        return (C*)(Environment::getEnvironment()->getGlobalMemory() + index + cellsIndex);
    }
    
    uint xdim, ydim;
    uint dimsIndex;
    uint quantitiesIndex;
    uint cellsIndex;
    uint index;
};

#endif
