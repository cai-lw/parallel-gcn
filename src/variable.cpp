#include "variable.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>

Variable::Variable(int size, bool requires_grad):
    data(size), grad(requires_grad ? size : 0) {}

int Variable::size() {
    return data.size();
}

void Variable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0 / (in_size + out_size));
    for(int i = 0; i < size(); i++)
        data[i] = (float(rand()) / RAND_MAX - 0.5) * range * 2;
}

void Variable::zero() {
    std::fill(data.begin(), data.end(), 0);
}

void Variable::zero_grad() {
    if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0);
}
