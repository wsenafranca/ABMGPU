#include "agent.cuh"
#include "../space/cell.cuh"

Agent::Agent() : cell(0), dead(0), toDie(0), children(0) {}

Agent::~Agent(){}

void Agent::inheritance(Agent *) {}

void Agent::move(Cell *cell) {
    nextCell = cell;
}

void Agent::die() {
    toDie = true;
}

void Agent::reproduce(uint children) {
    this->children = children;
}

