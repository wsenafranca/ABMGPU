#ifndef CONCURRENTQUEUE_H
#define CONCURRENTQUEUE_H

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

template<class T>
class ConcurrentQueue {
public:
    T pop() {
        std::unique_lock<std::mutex> mlock(mutex_);
        cond_.wait(mlock, [this]{ return this->m_stop || !this->queue_.empty(); });
        
        if(m_stop && this->queue_.empty()) return NULL;
        auto item = queue_.front();
        queue_.pop();
        return item;
    }
    void push(const T& item)
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.push(item);
        mlock.unlock();
        cond_.notify_one();
    }
    void start() {
        m_stop = false;
    }
    void stop() {
        std::unique_lock<std::mutex> mlock(mutex_);
        m_stop = true;
        cond_.notify_all();
    }
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool m_stop;
};

#endif
