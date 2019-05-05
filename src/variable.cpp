#include "variable.h"
#include "rand.h"
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
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < data.size(); i++)
        data[i] = (float(RAND()) / MY_RAND_MAX - 0.5) * range * 2;
}

void Variable::zero() {
    #ifdef SIMD
    #pragma omp parallel for simd schedule(static)
    #else
    #pragma omp parallel for schedule(static)
    #endif
    for(int i = 0; i < data.size(); i++)
        data[i] = 0;
}

void Variable::zero_grad() {
    #ifdef SIMD
    #pragma omp parallel for simd schedule(static)
    #else
    #pragma omp parallel for schedule(static)
    #endif
    for(int i = 0; i < grad.size(); i++)
        grad[i] = 0;
    #ifdef OMP
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < local_grad.size(); i++)
        #ifdef SIMD
        #pragma omp simd
        #endif
        for(int j = 0; j < local_grad[i].size(); j++)
            local_grad[i][j] = 0;
    #endif
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