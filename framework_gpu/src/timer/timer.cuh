#ifndef TIMER_CUH
#define TIMER_CUH

#include "../utils/concurrent_queue.h"

#include <map>
#include <set>
#include <vector>
#include <list>
#include <queue>

typedef std::multimap< uint, Event*>::iterator EventIteratorMap;

void work(uint id, cudaStream_t *streams, ConcurrentQueue<Event*> *tasks, bool *sleeping, uint *finished, bool *running) {
    setCurrentDevice();
    cudaEvent_t cuEv;
    cudaEventCreateWithFlags(&cuEv, cudaEventDisableTiming);
    while(true) {
        while(sleeping[id] && (*running)) std::this_thread::yield();
        if(!(*running)) break;
        Event *ev = tasks->pop();
        if(ev) {
            ev->getAction()->action(streams[id]);
            cudaEventRecord(cuEv, streams[id]);
            while(cudaEventQuery(cuEv)!=cudaSuccess) std::this_thread::yield();
            finished[id]++;
        }
        else sleeping[id] = true;
    }
    cudaEventDestroy(cuEv);
}

struct CompEvent{
    bool operator()(const Event *ev1, const Event *ev2) {
        if(ev1->getTime() > ev2->getTime()) 
            return true;
        else if(ev1->getTime() < ev2->getTime())
            return false;
        else
            return ev1->getPriority() > ev2->getPriority();
    }
};

class Timer{
public:
    Timer(const uint Workers) : numWorkers(Workers) {
        sleeping = new bool[numWorkers];
        finished = new uint[numWorkers];
        streams = new cudaStream_t[numWorkers];       
    }
    
    ~Timer() {
        delete [] sleeping;
        delete [] streams;
        delete [] finished;
    }
    
    uint getNumWorkers() const {return numWorkers;}
    
    void start() {
        tasks.start();
        running = true;
        
        wakeup();
        for(uint i = 0; i < numWorkers; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            workers.push_back(std::thread(work, i, streams, &tasks, sleeping, finished, &running));
        }
    }
    
    void stop() {
        tasks.stop();
        running = false;
        for(int i = 0; i < numWorkers; i++) {
            workers[i].join();
            cudaStreamDestroy(streams[i]);
        }
    }
    
    void execute(uint init, uint end, uint step = 1) {    
        currentTime = init;
        while(currentTime <= end) {
            cudaDeviceSynchronize();
            CHECK_ERROR
            memset(finished, 0, sizeof(uint)*numWorkers);
            
            std::list<Event*> nextEvents;
            uint numTasks = 0;
            int currentPriority;
            bool first = true;
            while(!events.empty()) {
                Event *ev = events.top();
                if(first) {
                    currentTime = ev->getTime();
                    currentPriority = ev->getPriority();
                    first = false;
                }
                if(ev->getTime() == currentTime && ev->getPriority() == currentPriority) {
                    numTasks++;
                    events.pop();
                    tasks.push(ev);
                    if(ev->getPeriod() > 0) {
                        nextEvents.push_back(ev);
                    }
                }
                else {
                    if(ev->getTime() != currentTime) currentTime+=step;
                    first = true;
                    break;
                }
            }

            while(sumFinished() != numTasks) {
                std::this_thread::yield();
            }
            
            for(Event *ev : nextEvents) {
                ev->next();
                addEvent(ev);
            }
            if(events.empty()) break;
        }
    }
    
    void createEvent(uint time, int priority, uint period, Action *action) {
        Event *ev = new Event(time, priority, period, action);
        addEvent(ev);
    }
    
    void addEvent(Event *ev) {
        events.push(ev);
    }
    
private:
    void wakeup() {
        memset(sleeping, 0, sizeof(bool)*numWorkers);
    }
    
    void sleep(uint n) {
        memset(sleeping, 1, sizeof(bool)*numWorkers);
    }
    
    uint sumFinished() {
        uint s = 0;
        for(uint i = 0; i < numWorkers; i++) s+= finished[i];
        return s;
    }
    
private:
    const uint numWorkers;
    std::priority_queue<Event*, std::vector<Event*>, CompEvent> events;
    uint currentTime;
    bool running;
    uint *finished;
    bool *sleeping;
    cudaStream_t *streams;
    ConcurrentQueue<Event*> tasks;
    std::vector<std::thread> workers;
};

#endif
