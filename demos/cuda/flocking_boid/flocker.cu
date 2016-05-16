#include <simgpu.cuh>

#define BOID_SIZE 2
#define NEIGHBOR 10
#define XDIM 2560
#define YDIM 2560
#define ITERATION 200
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
		double temp_value =  (pow(len,4) + 1);
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
	
	double rand_x = fma(cuRandom(boid->id),2.0f,-1.0f);//random[index*2]*2-1.0;
	double rand_y = fma(cuRandom(boid->id),2.0f,-1.0f);
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

int saveBMP(uint *map, unsigned int width, unsigned int height, const char *filename) {
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
            int b = BOID_SIZE;//cells[i].quantity*BOID_SIZE;
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
    Random obj(seed, POP_SIZE);
    
    Society<Boid> soc(POP_SIZE, POP_SIZE);
    CellularSpace<Cell> cs(XDIM, YDIM);    
    Neighborhood<Boid, Cell> nb(&soc, &cs, NEIGHBOR, NEIGHBOR);
    
    //uint *cells = (uint*)malloc(sizeof(uint)*XDIM*YDIM);
    
    soc.init();
    cs.init();
    
    placement(&soc, &cs);
    execute<init>(&soc);
    
    for(int i = 1; i <= ITERATION; i++) {
        //printf("%d\n", i);
        //cudaDeviceSynchronize();
        synchronize(&soc, &cs, &nb);
        execute<reset>(&soc);
        execute<collect>(&soc, &cs, &nb);
        execute<flocker>(&soc, &cs);
        //if(i%2==0) {
        //    cudaMemcpy(cells, cs.quantities, sizeof(uint)*XDIM*YDIM, cudaMemcpyDeviceToHost);
        //    static char filename[128];
        //    sprintf(filename, "img%d.bmp", i/2);
        //    saveBMP(cells, XDIM, YDIM, filename);
        //}
    }
    cudaDeviceSynchronize();
    
    //free(cells);
}

int main() {
    cudaSetDevice(1);
    
    size_t threads[] = {8, 64, 256, 512, 1024};
    
    long t;
    FILE *f = fopen("flockers_GPU.txt", "w");
    for(int i = 0; i < 5; i++) {
        THREADS = threads[i];
        POP_SIZE = 320000;
        fprintf(f, "Test %d: PopSize: %d (%lf)\n", i+1, POP_SIZE, (double)(double)POP_SIZE/(double)(XDIM*YDIM));
        printf("Test %d: PopSize: %d (%lf)\n", i+1, POP_SIZE, (double)(double)POP_SIZE/(double)(XDIM*YDIM));
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

