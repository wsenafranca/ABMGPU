#include "cell.cuh"

Cell::Cell() {}

Cell::~Cell() {}

uint Cell::getQuantity() const {
	return quantities[cid];
}
uint Cell::getX() const {
	return cid%(*xdim);
}
uint Cell::getY() const {
	return cid/(*xdim);
}

