#ifndef SIMULATION_CUH
#define SIMULATION_CUH

/**
* \class Simulation
* \brief The class that defines what  is needed to simulate an ABM in GPU.
**/
class Simulation {
public:
	/**
	* This method is called before the simulation starts.
	**/
	virtual void init();
	/**
	* This method is called in every iteration. It is used to provide the communication between the agents.
	**/
	virtual void communicate();
	/**
	* This method is called after the communicate phase. It is used to determine the actions of the agents in a single simulation step.
	* \param time The current time of the simulation.
	**/
	virtual void step(int time);
	/**
	* This method is called when all agents has stopped their actions. 
	* It is used to synchronize the information of all agents, performing the reproductions and removing the dead agents.
	**/
	virtual void synchronize();
	/**
	* This method is called after the simulation. It is used to cleanup the memory used during the simulation.
	**/
	virtual void finalize();
	/**
	* Execute the simulation.
	* \param time The quantity of time (simulation steps) the simulation will execute.
	**/
	void execute(int time);
	/**
	* Execute the simulation indefinitely.
	**/
	void execute();
	/**
	* Execute the simulation.
	* \param ini The start of the simulation.
	* \param end The end of the simulation.
	* \param step The amount of time between one step and next step.
	**/
	void execute(int ini, int end, int step);
	/**
	* Check if the simulation is running.
	**/
	bool isRunning() const;
private:
	void execute_step(int t);
	bool running;
};

#endif
