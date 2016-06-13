#include <simgpu.cuh>
#include <observer/image.cuh>

#define XDIM 256
#define YDIM 256

class MyCell : public Cell{
public:
    bool alive;
};

__global__ void transfer(MyCell *cells, bool *img, uint len) {
    uint i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < len) 
        img[i] = cells[i].alive;
}

template<class C>
class ObserverImage{
public:
    ObserverImage(CellularSpace<C> *cellSpace) : cs(cellSpace) {
        cudaMalloc(&d_img, sizeof(bool)*XDIM*YDIM);
        cudaMallocHost(&img, sizeof(bool)*XDIM*YDIM);
    }
    Event* observerEvent() {
        class ObserverAction : public Action {
        public:
            ObserverAction(CellularSpace<C> *cellSpace, bool *d_image, bool *image) 
                            : cs(cellSpace), d_img(d_image), img(image) {}
            void action(cudaStream_t &stream) {
                transfer<<<BLOCKS(cs->size), THREADS, 0, stream>>>(cs->getCellsDevice(), d_img, cs->size);
                cudaMemcpyAsync(img, d_img, sizeof(bool)*cs->size, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                char filename[256];
                uint t = getEvent()->getTime();
                sprintf(filename, "img%u.bmp", t);
                saveBMP(img, cs->xdim, cs->ydim, filename);
            }
            CellularSpace<C> *cs;
            bool *d_img, *img; 
        };
        return new Event(1, 0, 1, new ObserverAction(cs, d_img, img));
    }
    ~ObserverImage() {
        cudaFreeHost(img);
        cudaFree(d_img);
    }
    CellularSpace<C> *cs;
    bool *img, *d_img;
};

__device__ void init(Cell *c) {
    MyCell *cell = (MyCell*)c;
    cell->alive = cuRandom(ID) > 0.5f;
}

__device__ void life(Cell *c, Cell *cs, uint xdim, uint ydim) {
    MyCell *cell = (MyCell*)c;
    MyCell *past = (MyCell*)cs;
    
    bool alive = cell->alive;
        
    int x = cell->getX();
    int y = cell->getY();
    int count = 0;
    for(int i = -1; i < 2; i++) {
        for(int j = -1; j < 2; j++) {
            uint ncid = (y+i)*xdim + (x+j);
            if(cell->cid != ncid && ncid < xdim*ydim && past[ncid].alive) {
                count++;
            }
        }
    }
    
    if(alive)
        if(count < 2)
            cell->alive = false;
        else if(count <= 3)
            cell->alive = true;
        else
            cell->alive = false;
    else if(count == 3)
        cell->alive = true;
}

int main() {
    CURRENT_DEVICE = 1;
    
    Environment *env = new Environment();
    env->setRandomObject(new Random(1234, XDIM*YDIM));
    Timer *timer = new Timer(2);
    env->setTimer(timer);
    
    CellularSpace<MyCell> *cs = env->createCellularSpace<MyCell>(XDIM, YDIM);
    
    timer->addEvent(ExecuteEvent::execute<init>(cs, 0, 1, 0));
    timer->addEvent(ExecuteEvent::execute<life>(cs, 1, 1, 1));
    
    ObserverImage<MyCell> *obs = new ObserverImage<MyCell>(cs);
    timer->addEvent(obs->observerEvent());
    
    env->execute(100);
    
    delete obs;
    
    delete env;
    return 0;
}

