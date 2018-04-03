#include "Simulation.cuh"

#include <cstdio>
#include <chrono>
#include <algorithm>

void Simulation::init() {}

void Simulation::communicate() {}

void Simulation::step(int) {}

void Simulation::synchronize() {}

void Simulation::finalize() {}

void Simulation::execute(int time) {
	execute(1, time, 1);
}

void Simulation::execute() {
	init();
	synchronize();
	int t = 1;
	running = true;
	while(running) {
		execute_step(t++);
	}
	finalize();
}

void Simulation::execute(int ini, int end, int step) {
	size_t freeMem, totalMem, sysMem, usageMem=0;
    cudaMemGetInfo(&freeMem, &totalMem);
    sysMem = totalMem-freeMem;
    usageMem = std::max(usageMem, totalMem-freeMem-sysMem);
	
	auto t_begin = std::chrono::high_resolution_clock::now();
	init();		
	synchronize();
	running = true;
	
	for(int t = ini; t <= end; t+=step) {
		execute_step(t);
		
		cudaMemGetInfo(&freeMem, &totalMem);
    	usageMem = std::max(usageMem, totalMem-freeMem);
	}
	finalize();
	auto t_end = std::chrono::high_resolution_clock::now();
	printf("Time: %f seconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_begin).count()/1000.0f);
	printf("Mem: %.2lf/%.2lf MB (%.2lf)\n", usageMem/(double)(1024*1024), totalMem/(double)(1024*1024), ((double)usageMem/(double)totalMem));
}

bool Simulation::isRunning() const {
	return running;
}

void Simulation::execute_step(int t) {
	communicate();
	cudaError_t err;
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		fprintf(stderr, "Communication Error(%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(1);
	}
	
	step(t);
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		fprintf(stderr, "Logical Error(%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(1);
	}
	
	synchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		fprintf(stderr, "Synchronize Error(%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(1);
	}
}
