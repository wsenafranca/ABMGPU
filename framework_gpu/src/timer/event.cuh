#ifndef EVENT_CUH
#define EVENT_CUH

class Event;

class Action{
public:
    virtual void action(cudaStream_t&)=0;
    Event* getEvent() const {return ev;}
    Event *ev;
};

class Event{
public:
    Event(uint time, int priority, uint period, Action *action)  : 
            m_time(time), m_period(period), m_priority(priority), m_action(action) {
        m_action->ev = this;
    }
    
    Event(const Event &ev) : m_time(ev.m_time), m_period(ev.m_period), m_priority(ev.m_priority), m_action(ev.m_action){}
    
    virtual ~Event(){delete m_action;}
    
    void next() {
        m_time += m_period;
    }
    
    uint getTime() const {return m_time;}
    void setTime(uint time) {this->m_time = time;}
    
    uint getPeriod() const {return m_period;}
    void setPeriod(uint period) {this->m_period = period;}
    
    int getPriority() const {return m_priority;}
    void setPriority(int priority) {this->m_priority = priority;}
    
    Action* getAction() const {return m_action;}
    
private:
    uint m_time, m_period;
    int m_priority;
    Action* m_action;
};

#endif
