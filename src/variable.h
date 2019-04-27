#ifndef VARIABLE_H
#include <vector>

struct Variable {
    std::vector<float> data, grad;
    std::vector<std::vector<float>> local_grad;
    Variable(int size, bool requires_grad=true, bool thread_local_grad=false);
    void glorot(int in_size, int out_size);
    void zero();
    void zero_grad();
    void print(int col=0x7fffffff);
    float grad_norm();
};

#define VARIABLE_H
#endif