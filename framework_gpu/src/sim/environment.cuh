#ifndef ENVIRONMENT_CUH
#define ENVIRONMENT_CUH

#include "../utils/utils.cuh"

class Environment{
public:
    Environment() : size(0), d_mem(0) {}
    
    virtual ~Environment() {
        reset();
    }
    
    bool init() {
        cudaError_t err = cudaMalloc(&d_mem, size);
        printf("total memory: %lu\n", size);
        if(err != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(err));
            exit(1);
            return false;
        }
        return true;
    }
    
    uint alloc(size_t bytes) {
        uint index = size;
        size+=bytes;
        return index;
    }
    
    unsigned char* getGlobalMemory() {
        return d_mem;
    }
    
    void reset() {
        cudaFree(d_mem);
        d_mem = 0;
        size = 0;
        CHECK_ERROR
    }
    
    static Environment* getEnvironment() {
        if(!environment) environment = new Environment();
        return environment;
    }
    
private:
    size_t size;
    unsigned char *d_mem;
    static Environment* environment;
};

Environment *Environment::environment = 0;

#endif

