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
    double cons_x, cons_y, cohe_x, cohe_y, avoid_x, avoid_y;
    uint num_neighbor, num_non_dead;
};

__device__ void init(Agent *ag) {
    Boid *boid = (Boid*)ag;
    boid->x = boid->cell->getX();
    boid->y = boid->cell->getY();
}

__device__ void reset(Agent *ag) {
    Boid *boid = (Boid*)ag;
    boid->cons_x = 0;
    boid->cons_y = 0;
    boid->cohe_x = 0;
    boid->cohe_y = 0;
    boid->avoid_x = 0;
    boid->avoid_y = 0;
    boid->num_neighbor = 0;
    boid->num_non_dead = 0;
}

__device__ void collect(Agent *ag1, Agent *ag2) {
    Boid *boid1 = (Boid*)ag1;
    Boid *boid2 = (Boid*)ag2;
    //value for collect info
	int num_neighbor = 0;
	int num_non_dead = 0;
	double cons_x = 0.0;
	double cons_y = 0.0;
	double cohe_x = 0.0;
	double cohe_y = 0.0;
	double avoid_x = 0.0;
	double avoid_y = 0.0;
	
	double me_x = boid1->x;
	double me_y = boid1->y;
	
	double him_x = boid2->x;
	double him_y = boid2->y;
	double his_dx = boid2->dx;
	double his_dy = boid2->dy;
	
	double temp_tdx = tdx(me_x, him_x, XDIM);
	double temp_tdy = tdy(me_y, him_y, YDIM);
	
	double len = hypot(temp_tdx, temp_tdy);
	
	if(len <= NEIGHBOR) {
		double temp_value =  (powf(len,4) + 1);
		double temp_avoidance_x = temp_tdx/temp_value;
		double temp_avoidance_y = temp_tdy/temp_value;

		cons_x += his_dx;
		cons_y += his_dy;
		cohe_x += temp_tdx;
		cohe_y += temp_tdy;
        num_non_dead ++;

		avoid_x += temp_avoidance_x;
		avoid_y += temp_avoidance_y;
		num_neighbor++;
	}
	boid1->cons_x+=cons_x;
	boid1->cons_y+=cons_y;
	boid1->cohe_x+=cohe_x;
	boid1->cohe_y+=cohe_y;
	boid1->avoid_x+=avoid_x;
	boid1->avoid_y+=avoid_y;
	boid1->num_neighbor+=num_neighbor;
    boid1->num_non_dead+=num_non_dead;
}

__device__ void flocker(Agent *ag, Cell *cs, uint xdim, uint ydim) {
    
    Boid* boid = (Boid*)ag;
    
    double me_x = boid->x;
	double me_y = boid->y;
	double my_dx = boid->dx;
    double my_dy = boid->dy;
    
    int num_neighbor = boid->num_neighbor;
    int num_non_dead = boid->num_non_dead;
	double cons_x = boid->cons_x;
	double cons_y = boid->cons_y;
	double cohe_x = boid->cohe_x;
	double cohe_y = boid->cohe_y;
	double avoid_x = boid->avoid_x;
	double avoid_y = boid->avoid_y;
	
	double rand_x = fma(cuRandom(ID),2.0f,-1.0f);//random[index*2]*2-1.0;
	double rand_y = fma(cuRandom((ID)),2.0f,-1.0f);
	double rand_length = hypot(rand_x,rand_y);
	rand_x = 0.05*rand_x/rand_length;
	rand_y = 0.05*rand_y/rand_length;
    
    if (num_non_dead > 0) { 
        cohe_x = cohe_x/num_non_dead; 
        cohe_y = cohe_y/num_non_dead; 

        cons_x = cons_x/num_non_dead; 
        cons_y = cons_y/num_non_dead; 			
    }
    
	if(num_neighbor > 0) {
		avoid_x = avoid_x/num_neighbor;
		avoid_y = avoid_y/num_neighbor;
	}
	
	cohe_x = -cohe_x/100;
	cohe_y = -cohe_y/100;	
	avoid_x = 400*avoid_x;
	avoid_y = 400*avoid_y;
    
	my_dx = cohesion * cohe_x + avoidance * avoid_x + consistency* cons_x + randomness * rand_x + momentum *my_dx;
	my_dy = cohesion * cohe_y + avoidance * avoid_y + consistency* cons_y + randomness * rand_y + momentum *my_dy;
    
	double dis = hypot(my_dx, my_dy);
	
	if (dis>0) {
		double value = jump / dis;
		my_dx = my_dx *value;
		my_dy = my_dy *value;
	}
	
	double rx = _stx(me_x + my_dx, XDIM);
	double ry = _sty(me_y + my_dy, YDIM);
    int x = (int)roundf(rx);
    int y = (int)roundf(ry);
    if(x >= 0 && x < XDIM && y >= 0 && y < YDIM) {
        int newcid = y*xdim + x;
        boid->move(&cs[newcid]);
    }
    boid->x = rx;
    boid->y = ry;
    boid->dx = my_dx;
    boid->dy = my_dy;
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
        execute<reset>(&soc);
        execute<collect>(&soc, &cs, &nb);
        execute<flocker>(&soc, &cs);
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

