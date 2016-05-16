#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include <set>
#include <list>

size_t POP_SIZE;
#define XDIM 2560
#define YDIM 2560
#define NEIGHBOR 10
#define BOID_SIZE 2

#define ITERATION 200

#define DIST(x1, y1, x2, y2) (abs(x1-x2)+abs(y1-y2))

typedef unsigned int uint;
                                                                                                        
#define random() (rand()/(float)RAND_MAX)

static const double cohesion = 1.0;
static const double avoidance = 1.0;
static const double randomness = 1.0;
static const double consistency = 1.0;
static const double momentum = 1.0;
static const double deadFlockerProbability = 0.0;
static const double jump = 0.7;

double _stx(double x, double width) { 
    if (x >= 0) { 
        if (x < width) return x;
        return x - width;
    } 
    return x + width;
}
  
double tdx(double x1, double x2,double width) {

    if (fabs(x1-x2) <= width / 2)
        return x1 - x2;  

    double dx = _stx(x1,width) - _stx(x2,width);
    if (dx * 2 > width) return dx - width;
    if (dx * 2 < -width) return dx + width;
    return dx;
}   

double _sty( double y, double height) { 
    if (y >= 0) { 
        if (y < height) return y; 
        return y - height; 
    }
    return y + height;
}

double tdy(double y1, double y2, double height) {

    if (fabs(y1-y2) <= height / 2)
        return y1 - y2;  // no wraparounds  -- quick and dirty check

    double dy = _sty(y1,height) - _sty(y2,height);
    if (dy * 2 > height) return dy - height;
    if (dy * 2 < -height) return dy + height;
    return dy;
}

class Agent;
class Society;

class Cell{
public:
    Cell() {}
    
    std::set<Agent*> agents;
    
    uint x;
    uint y;
};

class CellularSpace{
public:
    CellularSpace(uint xDim, uint yDim):xdim(xDim), ydim(yDim) {
        cells  = new Cell[xdim*ydim];
        for(uint i = 0, k = 0; i < ydim; i++) {
            for(uint j = 0; j < xdim; j++, k++) {
                cells[k].x = j;
                cells[k].y = i;
            }
        }
    }
    
    ~CellularSpace() {
        delete [] cells;
    }
    
    uint xdim, ydim;
    Cell *cells;
};

class Agent{
public:
    Agent() : cell(0), nextCell(0), dx(0), dy(0){}
    
    void move(Cell *cell) {
        nextCell = cell;
    }
    
    void syncMove() {
        if(nextCell && nextCell != cell) {
            Cell *newCell = nextCell;
            Cell *oldCell = cell;
            if(oldCell) oldCell->agents.erase(this);
            newCell->agents.insert(this);
            cell = newCell;
        }
        nextCell = NULL;
    }
    
    Cell *cell, *nextCell;
    
    double x, y, dx, dy;
    double cons_x, cons_y, cohe_x, cohe_y, avoid_x, avoid_y;
    int num_neighbor, num_non_dead;
};

class Society{
public:
    Society(uint quantity) {
        for(uint i = 0; i < quantity; i++) {
            agents.push_back(new Agent());
        }
    }
    ~Society() {
        std::list<Agent*>::iterator it;
        for(it = agents.begin(); it != agents.end(); it++)
            delete *it;
    }
    uint getQuantity() const {
        return agents.size();
    }
    
    std::list<Agent*> agents;
};

void placement(Society *soc, CellularSpace *cs) {
    std::list<Agent*>::iterator it;
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        Agent *ag = *it;
        uint cellId = rand()%(cs->ydim*cs->xdim);
        ag->move(&cs->cells[cellId]);
        ag->syncMove();
        ag->x=cs->cells[cellId].x;
        ag->y=cs->cells[cellId].y;
    }
}

void update(Agent *ag) {    
    ag->cons_x = 0;
    ag->cons_y = 0;
    ag->cohe_x = 0;
    ag->cohe_y = 0;
    ag->avoid_x = 0;
    ag->avoid_y = 0;
    ag->num_neighbor = 0;
    ag->num_non_dead = 0;
    ag->syncMove();
}

void collectInfo(Agent *a1, Agent *a2) {
    
    //value for collect info
	int num_neighbor = 0;
	int num_non_dead = 0;
	double cons_x = 0.0;
	double cons_y = 0.0;
	double cohe_x = 0.0;
	double cohe_y = 0.0;
	double avoid_x = 0.0;
	double avoid_y = 0.0;
	
	double me_x = a1->x;
	double me_y = a1->y;
	
	double him_x = a2->x;
	double him_y = a2->y;
	double his_dx = a2->dx;
	double his_dy = a2->dy;
	
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
	a1->cons_x+=cons_x;
	a1->cons_y+=cons_y;
	a1->cohe_x+=cohe_x;
	a1->cohe_y+=cohe_y;
	a1->avoid_x+=avoid_x;
	a1->avoid_y+=avoid_y;
	a1->num_neighbor+=num_neighbor;
    a1->num_non_dead+=num_non_dead;
}

void flocker(Agent *ag, CellularSpace *cs, uint xdim, uint ydim) {    
    double me_x = ag->x;
	double me_y = ag->y;
	double my_dx = ag->dx;
    double my_dy = ag->dy;
    
    int num_neighbor = ag->num_neighbor;
    int num_non_dead = ag->num_non_dead;
	double cons_x = ag->cons_x;
	double cons_y = ag->cons_y;
	double cohe_x = ag->cohe_x;
	double cohe_y = ag->cohe_y;
	double avoid_x = ag->avoid_x;
	double avoid_y = ag->avoid_y;
	
	double rand_x = fma(random(),2.0f,-1.0f);//random[index*2]*2-1.0;
	double rand_y = fma(random(),2.0f,-1.0f);
	
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
    my_dx = avoidance * avoid_x + momentum *my_dx;
	my_dy = avoidance * avoid_y + momentum *my_dy;
       
	double dis = hypot(my_dx, my_dy);
	
	if (dis>0) {
		double value = jump / dis;
		my_dx = my_dx *value;
		my_dy = my_dy *value;
	}
	
	double dx = _stx(me_x + my_dx, xdim);
	double dy = _sty(me_y + my_dy, ydim);
    int x = (int)roundf(dx);
    int y = (int)roundf(dy);
    if(x >= 0 && x < (int)xdim && y >= 0 && y < (int)ydim) {
        uint newcid = y*xdim+x;
        ag->move(&cs->cells[newcid]);
        ag->x = dx;
        ag->y = dy;
        ag->dx = my_dx;
        ag->dy = my_dy;
    }
}

void execute(Society *soc, CellularSpace *cs) {
    std::list<Agent*>::iterator it;
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        update(*it);
    }
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        Agent *ag = *it;
        Cell *cell = ag->cell;
        int x = cell->x;
        int y = cell->y;
        for(int i = -NEIGHBOR; i <= NEIGHBOR; i++) {
            for(int j = -NEIGHBOR; j <= NEIGHBOR; j++) {
                int nx = x+j;
                int ny = y+i;
                if(nx >= 0 && nx < (int)cs->xdim && ny >= 0 && ny < (int)cs->ydim){
                    uint newcid = ny*cs->xdim+nx;
                    Cell *newCell = &cs->cells[newcid];
                    std::set<Agent*>::iterator aIt;
                    for(aIt = newCell->agents.begin(); aIt != newCell->agents.end(); aIt++) {
                        Agent *other = *aIt;
                        if(other != ag)
                            collectInfo(ag, other);
                    }
                }
            }
        }
    }
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        flocker(*it, cs, cs->xdim, cs->ydim);
    }
}

int saveBMP(CellularSpace *cs, unsigned int width, unsigned int height, const char *filename) {
    unsigned int filesize = 54 + 3*width*height;
    
    unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*width*height*3);
    
    for(unsigned int i = 0; i < height*width; i++) {
        img[i*3+2] = 255;
        img[i*3+1] = 255;
        img[i*3+0] = 255;
    }
    
    for(unsigned int i = 0; i < height*width; i++) {
        if(!cs->cells[i].agents.empty()) {
            //printf("%d\n", cells[i].quantity);
            int b = BOID_SIZE;//cells[i].quantity*BOID_SIZE;
            for(int j = -b/2; j <= b/2; j++) {
                for(int k = -b/2; k <= b/2; k++) {
                    int x = i/width;
                    int y = i%width;
                    int idx = (y+j)*width+(x+k);
                    if(idx >= 0 && idx < (int)(width*height)) {
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
    srand(seed);
    Society pop(POP_SIZE);
    CellularSpace cs(XDIM, YDIM);
    placement(&pop, &cs);
    puts("...");
    //saveBMP(&cs, XDIM, YDIM, "img0.bmp");
    //printf("%ldx%ld(%ld)\n", XDIM, YDIM, XDIM*YDIM);
    for(int i = 1; i <= ITERATION; i++) {
        //printf("%d\n", i);
        execute(&pop, &cs);
        //if(i%2==0) {
         //   static char filename[256];
         //   sprintf(filename, "images/img%d.bmp", i/2);
         //   saveBMP(&cs, XDIM, YDIM, filename);
        //}
    }
}

int main() {
    //size_t sizes[] = {320000, 640000, 1280000, 1600000, 2560000};
    size_t sizes[] = {320000, 1600000};
    
    //system("date");
    long t;
    FILE *f = fopen("flockers11_CPU.txt", "w");
    for(int i = 0; i < 2; i++) {
        POP_SIZE = sizes[i];
        fprintf(f, "Test %d: PopSize: %ld (%lf)\n", i+1, POP_SIZE, (double)(double)POP_SIZE/(double)(XDIM*YDIM));
        printf("Test %d: PopSize: %ld (%lf)\n", i+1, POP_SIZE, (double)(double)POP_SIZE/(double)(XDIM*YDIM));
        for(int j = 0; j < 5; j++) {
            t = clock();
            runModel(j*10);
            t = clock()-t;
            printf("elapsed time %lf secs\n", t/(double) CLOCKS_PER_SEC);
            fprintf(f, "elapsed time %lf secs\n", t/(double) CLOCKS_PER_SEC);
        }
    }
    
    //system("date");
    fclose(f);
	return 0;
}

