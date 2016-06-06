#ifndef NEIGHBORHOOD_CUH
#define NEIGHBORHOOD_CUH

#include "../space/cellularspace.cuh"
#include "../agents/society.cuh"

typedef texture<uint, cudaTextureType2D, cudaReadModeElementType> UIntTexture2D;

UIntTexture2D beginsRef;
UIntTexture2D endsRef;

template<class A, class C>
class Neighborhood {
public:
    Neighborhood(Society<A> *soc, CellularSpace<C> *cs, uint n, uint m) {
        this->n = n;
        this->m = m;
        neighborhoodXDim = m > 0 ? truncf(cs->xdim/m)+1 : cs->xdim;
        neighborhoodYDim = n > 0 ? truncf(cs->ydim/n)+1 : cs->ydim;
        uint globalMemory = 0;
        
        neighborhoodIndex = globalMemory;
        globalMemory += sizeof(uint)*soc->capacity;                           // alloc neighborhood
        
        inhabitedIndex = globalMemory;
        globalMemory += sizeof(uint)*soc->capacity;                         // alloc inhabited
        
        offsetIndex = globalMemory;
        globalMemory += sizeof(uint)*neighborhoodXDim*neighborhoodYDim;     // offset
        
        posIndex = globalMemory;
        globalMemory += sizeof(uint)*neighborhoodXDim*neighborhoodYDim;     // pos
        
        quantitiesIndex = globalMemory;
        globalMemory += sizeof(uint)*neighborhoodXDim*neighborhoodYDim;     // quantities
        
        index = Environment::getEnvironment()->alloc(globalMemory);
        
    }
    
    ~Neighborhood() {
        cudaFreeArray(beginsArray);
        cudaFreeArray(endsArray);
    }
    
    void init() {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint>();
        cudaMallocArray(&beginsArray,&channelDesc,neighborhoodXDim,neighborhoodYDim,cudaArraySurfaceLoadStore);
        cudaBindTextureToArray(beginsRef,beginsArray, channelDesc);
        
        channelDesc = cudaCreateChannelDesc<uint>();
        cudaMallocArray(&endsArray,&channelDesc,neighborhoodXDim,neighborhoodYDim,cudaArraySurfaceLoadStore);
        cudaBindTextureToArray(endsRef,endsArray, channelDesc);
    }
    
    uint* getNeighborhoodDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+neighborhoodIndex);
    }
    
    uint* getInhabitedDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+inhabitedIndex);
    }
    
    uint* getOffsetDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+offsetIndex);
    }
    
    uint* getPosDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+posIndex);
    }
    
    uint* getQuantitiesDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+quantitiesIndex);
    }
    
    void clear() {        
        uint bytes = sizeof(uint)*neighborhoodXDim*neighborhoodYDim+
                     sizeof(uint)*neighborhoodXDim*neighborhoodYDim+
                     sizeof(uint)*neighborhoodXDim*neighborhoodYDim;
        cudaMemset(getOffsetDevice(), 0, bytes);
    }
    
    void syncTexture() {
        cudaMemcpyToArray(beginsArray,0, 0, getOffsetDevice(), 
                          sizeof(uint)*neighborhoodXDim*neighborhoodYDim,cudaMemcpyDeviceToDevice);
        cudaMemcpyToArray(endsArray,0, 0, getQuantitiesDevice(), 
                          sizeof(uint)*neighborhoodXDim*neighborhoodYDim,cudaMemcpyDeviceToDevice);
    }
    
    uint neighborhoodIndex;
    uint inhabitedIndex;
    uint offsetIndex;
    uint posIndex;
    uint quantitiesIndex;
    uint index;
    uint n, m;
    uint neighborhoodXDim;
    uint neighborhoodYDim;
    
    cudaArray *beginsArray, *endsArray;
};

#endif

