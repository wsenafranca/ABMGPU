#include"framework_gpu/simgpu.cuh"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#define XDIM 128
#define YDIM 128

#define POP_SIZE truncf(XDIM*YDIM*0.1f)
#define ENERGY 40
#define GROWTH 4

#define ITERATION 1000

#define PASTURE 0
#define SOIL 1
#define PREDATOR 0
#define PREY 1
#define MAX_TARGETS 100
                                                                                              
class Soil : public Cell{
public:
    __device__ Soil() : cover(0), count(0), target(0){}
    
    int cover;
    int count;
    Agent *target;
};

class Entity : public Agent{
public:
    __device__ Entity() : numTargets(0), energy(ENERGY), nextEnergy(ENERGY), type(0) {
    }
    
    __device__ void init() {
    	type = (id < POP_SIZE*0.25f) ? PREDATOR : PREY;
    }
        
    __device__ void targetOn(Entity *other) {
        if(numTargets < MAX_TARGETS) {
            targets[numTargets++] = other;
            other->predator = this;
        }        
    }
    
    __device__ void targetOn() {
        Soil *cell = (Soil*)this->cell;
        cell->target = cell->target ? cell->target : this;        
    }
    
    __device__ void clone(Agent *p) {
    	Entity *parent = (Entity*)p;
    	type = parent->type;
    }
    
    __device__ void eat() {
        if(type==PREDATOR) {
            for(int i = 0; i < numTargets; i++) {
                Entity *prey = targets[i];
                if(prey->predator == this) {
                    nextEnergy += prey->energy/2;
                    prey->die();
                    break;
                }
            }
            numTargets = 0;
        }
        else {
        	Soil *cell = (Soil*)this->cell;
            if(cell->target==this && cell->cover == PASTURE) {
                cell->cover = SOIL;
                nextEnergy += 5;
            }
        }
    }
    
    Entity* targets[MAX_TARGETS];
    int numTargets;
    Entity* predator;
    int energy, nextEnergy;
    int type;
};

__device__ void regrowth(Cell *cell) {
	Soil *soil = (Soil*)cell;
    if(soil->cover == SOIL) {
        soil->count++;
        if(soil->count >= GROWTH){
            soil->cover = PASTURE;
            soil->count = 0;
            soil->target=NULL;
        }
    }
}

__constant__ int neighs[][2] = {{0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {0, 1}, {0, -1}};
__device__ void commonActions(Agent *agent, Cell *cs, uint xdim, uint ydim) {
	Entity *ag = (Entity*)agent;
	Soil *cells = (Soil*)cs;
    ag->nextEnergy--;
    if(ag->energy <= 0) {
        ag->die();
        return;
    }
    
    Soil *cell = (Soil*)ag->cell;
    
    int pos = cuRand(ag->id, 8)%8;
    int rx = neighs[pos][0];
    int ry = neighs[pos][1];
    
    size_t newX = cell->getX() + rx;
    size_t newY = cell->getY() + ry;

    if(newX < xdim && newY < ydim) {
        ag->move(&cells[newY*xdim+newX]);
    }
    
    if((ag->type == PREY && ag->energy >= 50) || (ag->type == PREDATOR && ag->energy >= 50)) {
        ag->nextEnergy /= 2;
        ag->reproduce(1);
    }
    
    if(ag->type == PREY)
    	ag->targetOn();
}

__device__ void eat(Agent *agent) {
	Entity *ag = (Entity*)agent;
	ag->eat();
}

__device__ void hunt(Agent *agent1, Agent *agent2) {
	Entity *ag = (Entity*)agent1;
	Entity *other = (Entity*)agent2;
	if(ag->type == PREDATOR && (other->type == PREY || cuRandom(ag->id) < 0.1))
        ag->targetOn(other);
}

__device__ void reset(Agent *agent) {
	Entity *ag = (Entity*)agent;
	ag->energy = ag->nextEnergy;
}

template<class A>
__global__ void count(A *agents, uint size, uint *numPredators, uint *numPreys) {
	uint i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i < size && !agents[i].dead) {
		if(agents[i].type == PREDATOR)
			atomicAdd(numPredators, 1);
		else
			atomicAdd(numPreys, 1);
	}	
}

template<class A>
void count(Society<A> *soc, uint *numPredators, uint *numPreys) {
	uint *d_predators, *d_preys;
	cudaMalloc(&d_predators, sizeof(uint));
	cudaMalloc(&d_preys, sizeof(uint));
	cudaMemset(d_predators, 0, sizeof(uint));
	cudaMemset(d_preys, 0, sizeof(uint));
	uint blocks = BLOCKS(soc->size);
	count<<<blocks, THREADS>>>(soc->agents, soc->size, d_predators, d_preys);
	cudaMemcpy(numPredators, d_predators, sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(numPreys, d_preys, sizeof(uint), cudaMemcpyDeviceToHost);
	cudaFree(d_predators);
	cudaFree(d_preys);
}

int saveBMP(unsigned int *map, unsigned int width, unsigned int height, const char *filename) {
    unsigned int filesize = 54 + 3*width*height;
    
    unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*width*height*3);
    
    for(unsigned int i = 0; i < height*width; i++) {
        img[i*3+2] = 255;
        img[i*3+1] = 255;
        img[i*3+0] = 255;
    }
    
    for(unsigned int i = 0; i < height*width; i++) {
        if(map[i] > 0) {
            //printf("%d\n", cells[i].quantity);
            int b = 2;//cells[i].quantity*BOID_SIZE;
            for(int j = -b/2; j <= b/2; j++) {
                for(int k = -b/2; k <= b/2; k++) {
                    int x = i/width;
                    int y = i%width;
                    int idx = (y+j)*width+(x+k);
                    if(idx >= 0 && idx < width*height) {
                        img[idx*3+2] = 0;
                        img[idx*3+1] = 0;
                        img[idx*3+0] = 0;
                    }
                }
            }
        }
    }
    
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};
    
    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(width);
    bmpinfoheader[ 5] = (unsigned char)(width>> 8);
    bmpinfoheader[ 6] = (unsigned char)(width>>16);
    bmpinfoheader[ 7] = (unsigned char)(width>>24);
    bmpinfoheader[ 8] = (unsigned char)(height);
    bmpinfoheader[ 9] = (unsigned char)(height>> 8);
    bmpinfoheader[10] = (unsigned char)(height>>16);
    bmpinfoheader[11] = (unsigned char)(height>>24);
    
    FILE *f = fopen(filename,"wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(unsigned i=0; i<height; i++) {
        fwrite(img+(width*(height-i-1)*3),3,width,f);
        fwrite(bmppad,1,(4-(width*3)%4)%4,f);
    }
    fclose(f);
    
    free(img);
    return 0;
}

void runModel(long seed) {
	uint capacity = XDIM*YDIM;
	Random obj(seed, capacity);
    
    Society<Entity> soc(POP_SIZE, capacity);
    CellularSpace<Soil> cs(XDIM, YDIM);    
    Neighborhood<Entity, Soil> nb(&soc, &cs, 0, 0);
    
    soc.init();
    cs.init();
    
    placement(&soc, &cs);
    
    //uint *cells = (uint*)malloc(sizeof(uint)*XDIM*YDIM);
    
    uint numPredators, numPreys;
    for(int i = 1; i <= ITERATION; i++) {
    	count(&soc, &numPredators, &numPreys);
    	cudaDeviceSynchronize();
        //printf("%d %d %d %d\n", i, numPredators+numPreys, numPredators, numPreys);
        synchronize(&soc, &cs, &nb);
        execute<reset>(&soc);
        execute<hunt>(&soc, &cs, &nb);
        execute<commonActions>(&soc, &cs);
        execute<eat>(&soc);
        execute<regrowth>(&cs);
        
        //if(i%20==0) {
        //    cudaMemcpy(cells, cs.quantities, sizeof(uint)*XDIM*YDIM, cudaMemcpyDeviceToHost);
        //    static char filename[128];
        //    sprintf(filename, "img%d.bmp", i/20);
        //    saveBMP(cells, XDIM, YDIM, filename);
        //}
    }
    cudaDeviceSynchronize();
    //free(cells);
}

int main() {
	cudaSetDevice(1);
    
    //size_t sizes[] = {128, 256, 512, 768, 960};
    
    long t;
    FILE *f = fopen("preys_GPU.txt", "w");
    for(int i = 0; i < 1; i++) {
        //XDIM = sizes[i];
        //YDIM = sizes[i];
        fprintf(f, "Test %d: Dim: %d (%lf)\n", i+1, XDIM, (double)(double)POP_SIZE/(double)(XDIM*YDIM));
        printf("Test %d: Dim: %d (%lf)\n", i+1, XDIM, (double)(double)POP_SIZE/(double)(XDIM*YDIM));
        for(int j = 0; j < 5; j++) {
        	cudaDeviceReset();
        	        	
            t = clock();
            runModel(j*10);
            t = clock()-t;
            printf("elapsed time %lf secs\n", t/(double) CLOCKS_PER_SEC);
            fprintf(f, "elapsed time %lf secs\n", t/(double) CLOCKS_PER_SEC);
        }
    }

    fclose(f);
	return 0;
    
}

