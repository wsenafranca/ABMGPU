#include <simgpu.cuh>

//#define OBSERVER

#ifdef OBSERVER
#include <observer/image.cuh>
#define XDIM 512
#define YDIM 512
#define ITERATION 1000

#else

#define XDIM 2560
#define YDIM 2560
#define ITERATION 200

#endif

#define BOID_SIZE 2
#define NEIGHBOR 10
uint POP_SIZE;

// constants

const double cohesion = 1.0;
const double avoidance = 1.0;
const double randomness = 1.0;
const double consistency = 1.0;
const double momentum = 1.0;
const double jump = 0.7;

// utils

__device__ double _stx(double x, double width) { 
    if (x >= 0) { 
        if (x < width) return x;
        return x - width;
    } 
    return x + width;
}
  
__device__  double tdx(double x1, double x2,double width) {

    if (fabs(x1-x2) <= width / 2)
        return x1 - x2;  

    double dx = _stx(x1,width) - _stx(x2,width);
    if (dx * 2 > width) return dx - width;
    if (dx * 2 < -width) return dx + width;
    return dx;
}   

__device__  double _sty( double y, double height) { 
    if (y >= 0) { 
        if (y < height) return y; 
        return y - height; 
    }
    return y + height;
}

__device__  double tdy(double y1, double y2, double height) {

    if (fabs(y1-y2) <= height / 2)
        return y1 - y2;  // no wraparounds  -- quick and dirty check

    double dy = _sty(y1,height) - _sty(y2,height);
    if (dy * 2 > height) return dy - height;
    if (dy * 2 < -height) return dy + height;
    return dy;
}

// behaviour

class Boid : public Agent{
public:
    __device__ Boid() : dx(0), dy(0) {}
    
    double dx, dy;
    double x, y;
};

__device__ void init(Agent *ag) {
    Boid *boid = (Boid*)ag;
    boid->x = boid->cell->getX();
    boid->y = boid->cell->getY();
}

namespace Iterator {

class CellIterator{
public:
    __device__ CellIterator() {}
    
    __device__ CellIterator(const CellIterator &it) : p(it.p) {}
    
    __device__ CellIterator(uint *ptr) : p(ptr) {}
    
    __device__ CellIterator& operator++() {p++;return *this;}
    
    __device__ CellIterator operator++(int) {CellIterator tmp(*this); operator++(); return tmp;}
    
    __device__ const CellIterator& operator=(const CellIterator &it) {p = it.p; return *this;}
    
    __device__ bool operator==(const CellIterator& rhs) {return p==rhs.p;}
    
    __device__ bool operator!=(const CellIterator& rhs) {return p!=rhs.p;}
    
    __device__ uint& operator*() {return *p;}
    
private:
    uint *p;
};
}

namespace Collection{

class Neighborhood{
public:
    __device__ Neighborhood(Agent *ag, uint *neighborhood, uint n, uint m, uint nxdim, uint nydim) {
        uint2 pos = ag->cell->getPos();
        int x = truncf(pos.x/m);
        int y = truncf(pos.y/n);
        neighs = 0;
        for(int ny = y-1; ny <= y+1; ny++) {
            for(int nx = x-1; nx <= x+1; nx++) {
                if(nx >= 0 && nx < nxdim && ny >= 0 && ny < nydim) {
                    uint begin = tex2D(beginsRef, nx, ny);
                    uint end = tex2D(endsRef, nx, ny);
                    if(end > 0) {
                        begins[neighs] = Iterator::CellIterator(&neighborhood[begin]);
                        ends[neighs] = Iterator::CellIterator(&neighborhood[end]);
                        neighs++;
                    }
                }
            }
        }
    }
    __device__ ~Neighborhood() {
    }
    __device__ const Iterator::CellIterator& begin(uint idx) {
        return begins[idx];
    }
    __device__ const Iterator::CellIterator& end(uint idx) {
        return ends[idx];
    }
    Iterator::CellIterator begins[9];
    Iterator::CellIterator ends[9];
    uint neighs;
};

}

#define forEachNeighborhood(ag, nb) \
    Collection::Neighborhood nb(ag, neighborhood, n, m, nxdim, nydim);\
    Iterator::CellIterator it; \
    uint neighId; \
    for(uint i = 0; i < nb.neighs; i++)  \
        for(it = nb.begin(i), neighId = ((&past[*(it)] == ag) ? *(++it) : (*it)); it != nb.end(i); \
                              neighId = ((&past[*(++it)] == ag) ? *(++it) : (*it))) \
    
template<class A, class C>
__global__ void flockers(A *agents, A *past, uint size, C* cells, uint *neighborhood, uint n, uint m, uint nxdim, uint nydim) {
    uint idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size) {
        A *ag = &agents[idx];
        //value for collect info
		int num_neighbor = 0;
		int num_non_dead = 0;
		double cons_x = 0.0;
		double cons_y = 0.0;
		double cohe_x = 0.0;
		double cohe_y = 0.0;
		double avoid_x = 0.0;
		double avoid_y = 0.0;
		
		double	me_x = ag->x;
        double	me_y = ag->y;
        double 	old_dx = ag->dx;
        double  old_dy = ag->dy;
        
        double	him_x = 0.0;
        double	him_y = 0.0;
        double  his_dx = 0.0;
        double  his_dy = 0.0;
        int isdead = 0;
		
        double temp_tdx = XDIM;
        double temp_tdy = YDIM;
        double len = 0.0;

        Collection::Neighborhood nb(ag, neighborhood, n, m, nxdim, nydim);
        Iterator::CellIterator it;
        
        for(uint i = 0; i < nb.neighs; i++) {
            for(it = nb.begin(i); it != nb.end(i); ++it) {
                A *ag2 = &past[*it];
                if(ag == ag2) continue;
                
                him_x = ag2->x;
                him_y = ag2->y;
                his_dx = ag2->dx;
                his_dy = ag2->dy;

                temp_tdx =  tdx(me_x,him_x,XDIM);
                temp_tdy =  tdy(me_y,him_y,YDIM);

                len = hypot(temp_tdx,temp_tdy);
                if(len <= NEIGHBOR)
                {
                    double temp_value =  (powf(len,4) + 1);
	                double temp_avoidance_x = temp_tdx/temp_value;
	                double temp_avoidance_y = temp_tdy/temp_value;

	                if(isdead==0)
	                {
		                cons_x += his_dx;
		                cons_y += his_dy;
		                cohe_x += temp_tdx;
		                cohe_y += temp_tdy;
		                num_non_dead ++;
	                }

	                avoid_x += temp_avoidance_x;
	                avoid_y += temp_avoidance_y;
	                num_neighbor++;
                }
            }
        }
        
        double rand_x = fma (cuRandom(idx),2.0,-1.0);//random[index*2]*2-1.0;
		double rand_y = fma (cuRandom(idx),2.0,-1.0);
		
		double rand_length = hypot(rand_x,rand_y);
		rand_x = 0.05*rand_x/rand_length;
		rand_y = 0.05*rand_y/rand_length;
		
		if (num_non_dead > 0)
		{ 
			cohe_x = cohe_x/num_non_dead; 
			cohe_y = cohe_y/num_non_dead; 

			cons_x = cons_x/num_non_dead; 
			cons_y = cons_y/num_non_dead; 			
		}

		if(num_neighbor > 0)
		{
			avoid_x = avoid_x/num_neighbor;
			avoid_y = avoid_y/num_neighbor;
		}

		cohe_x = -cohe_x/10;
		cohe_y = -cohe_y/10;	
		avoid_x = 400*avoid_x;
		avoid_y = 400*avoid_y;

		double my_dx = cohesion * cohe_x + avoidance * avoid_x + consistency* cons_x + randomness * rand_x + momentum *old_dx; 
		double my_dy = cohesion * cohe_y + avoidance * avoid_y + consistency* cons_y + randomness * rand_y + momentum *old_dy;
           
		double dis = hypot(my_dx,my_dy);

		if (dis>0)
		{
				double value = jump / dis;
				my_dx = my_dx *value;
				my_dy = my_dy *value;
		}
		
		double rx = _stx(me_x + my_dx, XDIM);
	    double ry = _sty(me_y + my_dy, YDIM);
        int cx = (int)roundf(rx);
        int cy = (int)roundf(ry);
        if(cx >= 0 && cx < XDIM && cy >= 0 && cy < YDIM) {
            int newcid = cy*XDIM + cx;
            ag->move(&cells[newcid]);
        }
        ag->x = rx;
        ag->y = ry;
        ag->dx = my_dx;
        ag->dy = my_dy;
    }
}

#ifdef OBSERVER

template<class A>
__global__ void count(A *agents, uint size, uint *map) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        map[agents[i].cell->cid] = 1;
    }
}

template<class A>
void draw(uint i, Society<A> *soc, uint *map) {
    uint *d_map;
    cudaMalloc(&d_map, sizeof(uint)*XDIM*YDIM);
    cudaMemset(d_map, 0, sizeof(uint)*XDIM*YDIM);
    
    uint blocks = BLOCKS(soc->size);
    count<<<blocks, THREADS>>>(soc->getAgentsDevice(), soc->size, d_map);
    
    cudaMemcpy(map, d_map, sizeof(uint)*XDIM*YDIM, cudaMemcpyDeviceToHost);
    
    static char filename[128];
    sprintf(filename, "img%d.bmp", i);
    saveBMP(map, XDIM, YDIM, filename);
    
    cudaFree(d_map);
}

#endif

void run() {
    Random::randomObj = new Random(1234, POP_SIZE);
    
    Society<Boid> soc(POP_SIZE, POP_SIZE);
    CellularSpace<Cell> cs(XDIM, YDIM);    
    Neighborhood<Boid, Cell> nb(&soc, &cs, NEIGHBOR, NEIGHBOR);
    
    Environment::getEnvironment()->init();
    
    soc.init();
    cs.init();
    nb.init();
    
    placement(&soc, &cs);
    execute<init>(&soc);
    
#ifdef OBSERVER
    uint *map = (uint*)malloc(sizeof(uint)*XDIM*YDIM);
#endif
    
    for(int i = 1; i <= ITERATION; i++) {
        synchronize(&soc, &cs, &nb);
        flockers<<<BLOCKS(soc.size), THREADS>>>(soc.getAgentsDevice(), soc.getPastDevice(), soc.size, cs.getCellsDevice(), 
                 nb.getNeighborhoodDevice(), nb.n, nb.m, nb.neighborhoodXDim, nb.neighborhoodYDim);
#ifdef OBSERVER
        if(i % 10 == 0)
            draw(i/10, &soc, map);
#endif
    }
    
#ifdef OBSERVER
    free(map);
#endif
    
    cudaDeviceSynchronize();
    Environment::getEnvironment()->reset(); // temp
}

int main() {
    cudaSetDevice(1);
    
    uint pops[] = {320000, 640000, 1280000, 1600000, 2560000};
    for(int i = 0; i < 5; i++) {
        POP_SIZE = pops[i];
        
        long t = clock();
        run();
        t = clock()-t;
        printf("%lf\n", t/(double) CLOCKS_PER_SEC);
    }
    
    delete Environment::getEnvironment(); // temp
    
	return 0;
}

