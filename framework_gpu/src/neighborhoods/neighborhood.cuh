#ifndef NEIGHBORHOOD_CUH
#define NEIGHBORHOOD_CUH

typedef texture<uint, cudaTextureType2D, cudaReadModeElementType> UIntTexture2D;

UIntTexture2D beginsRef;
UIntTexture2D endsRef;

template<class A, class C>
class Neighborhood : public Collection{
public:
    Neighborhood(Society<A> *society, CellularSpace<C> *cellSpace, uint N, uint M) 
                    : Collection(0, 0), soc(society), cs(cellSpace), n(N), m(M)
    {
        neighborhoodXDim = m > 0 ? truncf(cs->xdim/m)+1 : cs->xdim;
        neighborhoodYDim = n > 0 ? truncf(cs->ydim/n)+1 : cs->ydim;
        
        neighborhoodIndex = alloc(sizeof(uint)*soc->capacity); // alloc neighborhood
        
        inhabitedIndex = alloc(sizeof(uint)*soc->capacity); // alloc inhabited
        
        offsetIndex = alloc(sizeof(uint)*neighborhoodXDim*neighborhoodYDim); // offset;
        
        posIndex = alloc(sizeof(uint)*neighborhoodXDim*neighborhoodYDim); // pos
        
        quantitiesIndex = alloc(sizeof(uint)*neighborhoodXDim*neighborhoodYDim); // quantities        
    }
    
    ~Neighborhood() {
        cudaFreeArray(beginsArray);
        cudaFreeArray(endsArray);
    }
    
    Event* initializeEvent() {
        class Init : public Action{
        public:
            Init(Neighborhood<A, C> *neighborhood) : nb(neighborhood) {}
            void action(cudaStream_t &stream) {
                cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint>();
                cudaMallocArray(&nb->beginsArray, &channelDesc, nb->neighborhoodXDim, nb->neighborhoodYDim, cudaArraySurfaceLoadStore);
                cudaBindTextureToArray(beginsRef, nb->beginsArray, channelDesc);
                
                channelDesc = cudaCreateChannelDesc<uint>();
                cudaMallocArray(&nb->endsArray,&channelDesc, nb->neighborhoodXDim, nb->neighborhoodYDim, cudaArraySurfaceLoadStore);
                cudaBindTextureToArray(endsRef, nb->endsArray, channelDesc);
            }
            Neighborhood<A, C> *nb;
        };
        return new Event(0, COLLECTION_PRIORITY, 0, new Init(this));
    }
    
    uint* getNeighborhoodDevice() {
        return (uint*)(getMemPtrDevice()+getIndexOnMemoryDevice()+neighborhoodIndex);
    }
    
    uint* getInhabitedDevice() {
        return (uint*)(getMemPtrDevice()+getIndexOnMemoryDevice()+inhabitedIndex);
    }
    
    uint* getOffsetDevice() {
        return (uint*)(getMemPtrDevice()+getIndexOnMemoryDevice()+offsetIndex);
    }
    
    uint* getPosDevice() {
        return (uint*)(getMemPtrDevice()+getIndexOnMemoryDevice()+posIndex);
    }
    
    uint* getQuantitiesDevice() {
        return (uint*)(getMemPtrDevice()+getIndexOnMemoryDevice()+quantitiesIndex);
    }
    
    void clear() {        
        uint bytes = sizeof(uint)*neighborhoodXDim*neighborhoodYDim+
                     sizeof(uint)*neighborhoodXDim*neighborhoodYDim+
                     sizeof(uint)*neighborhoodXDim*neighborhoodYDim;
        cudaMemset(getOffsetDevice(), 0, bytes);
    }
    
    void syncTexture(cudaStream_t &stream) {
        cudaMemcpyToArrayAsync(beginsArray,0, 0, getOffsetDevice(), 
                               sizeof(uint)*neighborhoodXDim*neighborhoodYDim,cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyToArrayAsync(endsArray,0, 0, getQuantitiesDevice(), 
                               sizeof(uint)*neighborhoodXDim*neighborhoodYDim,cudaMemcpyDeviceToDevice, stream);
    }
    
    Society<A> *soc;
    CellularSpace<C> *cs;
    
    uint neighborhoodIndex;
    uint inhabitedIndex;
    uint offsetIndex;
    uint posIndex;
    uint quantitiesIndex;
    uint n, m;
    uint neighborhoodXDim;
    uint neighborhoodYDim;
    
    cudaArray *beginsArray, *endsArray;
};

#endif

