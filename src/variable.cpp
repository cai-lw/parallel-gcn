#include "variable.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

Variable::Variable(int size, bool requires_grad=true) {
    data = new float[size];
    grad = nullptr;
    this->size = size;
    if (requires_grad) 
        grad = new float[size];
}

void Variable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0 / (in_size + out_size));
    for(int i = 0; i < size; i++)
        data[i] = (float(rand()) / RAND_MAX - 0.5) * range * 2;
}

void Variable::zero() {
    memset(data, 0, size * sizeof(float));
}

void Variable::zero_grad() {
    if (grad) memset(grad, 0, size * sizeof(float));
}

Variable::~Variable(){
    delete[] data;
    if (grad)
        delete[] grad;
}
