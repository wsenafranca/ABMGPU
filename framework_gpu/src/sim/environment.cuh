#ifndef ENVIRONMENT_CUH
#define ENVIRONMENT_CUH

#include "../utils/utils.cuh"
#include <list>

class Environment{
public:
    Environment(uint workers = 1) : size(0), d_mem(0), randomObj(0) {
        setCurrentDevice();
        timer = new Timer(workers);
    }
    
    virtual ~Environment() {
        clear();
        delete randomObj;
        delete timer;
    }
    
    bool init() {
        cudaError_t err = cudaMalloc(&d_mem, size); // alloc global memory
        printf("total memory: %lu\n", size);
        if(err != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(err));
            exit(1);
            return false;
        }
        for(Collection *c : collections) 
            c->setMemPtrDevice(getMemPtrDevice()); // set memory to collections
        if(!randomObj) 
            randomObj = new Random(0, 10000); // create default random
        return true;
    }
    
    void execute(uint end, uint step=1) {
        init();
        timer->addEvent(randomObj->generateRandomEvent()); // schedulling random generate
        timer->start();
        timer->execute(0, end, step);
        timer->stop();
    }
    
    uint alloc(size_t bytes) {
        uint index = size;
        size+=bytes;
        return index;
    }
    
    Timer* getTimer() const {
        return timer;
    }
    
    void setTimer(Timer *timer) {
        if(this->timer) delete this->timer;
        this->timer = timer;
    }
    
    template<class A> 
    Society<A>* createSociety(uint size) {
        return createSociety<A>(size, size);
    }
    
    template<class A> 
    Society<A>* createSociety(uint size, uint capacity) {
        Society<A> *soc = new Society<A>(size, capacity);
        soc->setIndexOnMemoryDevice(alloc(soc->getBytes()));
        timer->addEvent(soc->initializeEvent());
        timer->createEvent(1, SYNCHRONIZE_PRIORITY, 1, new SynchronizeAgentAction<A>(soc));
        collections.push_back(soc);
        return soc;
    }
        
    template<class C>
    CellularSpace<C>* createCellularSpace(uint xdim, uint ydim) {
        CellularSpace<C> *cs = new CellularSpace<C>(xdim, ydim);
        cs->setIndexOnMemoryDevice(alloc(cs->getBytes()));
        timer->addEvent(cs->initializeEvent());
        timer->createEvent(1, SYNCHRONIZE_PRIORITY, 1, new SynchronizeSpaceAction<C>(cs));
        collections.push_back(cs);
        return cs;
    }
    
    template<class A, class C>
    Neighborhood<A, C>* createNeighborhood(Society<A> *soc, CellularSpace<C> *cs, uint n, uint m) {
        Neighborhood<A, C> *nb = new Neighborhood<A, C>(soc, cs, n, m);
        nb->setIndexOnMemoryDevice(alloc(nb->getBytes()));
        timer->addEvent(nb->initializeEvent());
        timer->createEvent(1, INDEXING_PRIORITY, 1, new IndexingAgentAction<A, C>(soc, cs, nb));
        collections.push_back(nb);
        return nb;
    }
    
    template<class A, class C>
    void createPlacement(Society<A> *soc, CellularSpace<C> *cs) {
        class CreatePlacement : public Action{
        public:
            CreatePlacement(Society<A> *society, CellularSpace<C> *cellularspace) 
                                : soc(society), cs(cellularspace) {}
            void action(cudaStream_t &stream) {
                uint blocks = BLOCKS(soc->size);
                placementKernel<<<blocks, THREADS, 0, stream>>>(soc->getAgentsDevice(), soc->size, 
                                            cs->getCellsDevice(), cs->xdim, cs->ydim);
            }
            Society<A> *soc;
            CellularSpace<C> *cs;
        };
        timer->addEvent(new Event(0, PLACEMENT_PRIORITY, 0, new CreatePlacement(soc, cs)));
    }
    
    unsigned char* getMemPtrDevice() {
        return d_mem;
    }
    
    void clear() {
        if(d_mem) cudaFree(d_mem);
        d_mem = 0;
        size = 0;
        CHECK_ERROR
        for(Collection *c : collections) 
            delete c;
        collections.clear();
    }
    
    Random* getRandomObject() {
        return randomObj;
    }
    
    void setRandomObject(Random *randomObj) {
        if(this->randomObj) delete this->randomObj;
        this->randomObj = randomObj;
    }
    
private:
    size_t size;
    unsigned char *d_mem;
    Random *randomObj;
    Timer *timer;
    std::list<Collection*> collections;
};

template<>
CellularSpace<Cell>* Environment::createCellularSpace(uint xdim, uint ydim) {
    CellularSpace<Cell> *cs = new CellularSpace<Cell>(xdim, ydim, false);
    cs->setIndexOnMemoryDevice(alloc(cs->getBytes()));
    timer->addEvent(cs->initializeEvent());
    collections.push_back(cs);
    return cs;
}

#endif

