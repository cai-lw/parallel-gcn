#ifndef VARIABLE_H
#include <vector>

struct Variable {
    float* data;
    float* grad;
    int size;
    Variable(int size, bool requires_grad=true);
    void glorot(int in_size, int out_size);
    void zero();
    void zero_grad();
    ~Variable();
};

#define VARIABLE_H
#endif