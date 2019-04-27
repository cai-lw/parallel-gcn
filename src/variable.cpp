#include "variable.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#ifdef OMP
#include <omp.h>
#endif

// https://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

Variable::Variable(int size, bool requires_grad, bool thread_local_grad):
    data(size), grad(requires_grad ? size : 0)
    #ifdef OMP
    ,local_grad(thread_local_grad ? omp_thread_count() : 0, std::vector<float>(size))
    #endif
    {}

void Variable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f / (in_size + out_size));
    for(int i = 0; i < data.size(); i++)
        data[i] = (float(rand()) / RAND_MAX - 0.5) * range * 2;
}

void Variable::zero() {
    std::fill(data.begin(), data.end(), 0);
}

void Variable::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0);
    for(auto &v: local_grad)
        std::fill(v.begin(), v.end(), 0);
}

void Variable::print(int col) {
    int count = 0;
    for(float x: data) {
        printf("%.4f ", x);
        count++;
        if(count % col == 0) printf("\n");
    }
}

float Variable::grad_norm() {
    float norm = 0;
    for(float x: grad) norm += x * x;
    return sqrtf(norm);
}