#include "variable.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>

Variable::Variable(int size, bool requires_grad):
    data(size), grad(requires_grad ? size : 0) {}

int Variable::size() {
    return data.size();
}

void Variable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f / (in_size + out_size));
    for(int i = 0; i < size(); i++)
        data[i] = (float(rand()) / RAND_MAX - 0.5) * range * 2;
}

void Variable::zero() {
    std::fill(data.begin(), data.end(), 0);
}

void Variable::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0);
}

void Variable::print(int col) {
    int count = 0;
    for(float x: data) {
        printf("%.4f ", x);
        count++;
        if(count % col == 0) printf("\n");
    }
}