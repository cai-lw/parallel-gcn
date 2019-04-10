#ifndef OPTIM_H
#include <vector>
#include "variable.h"

struct AdamParams {
    float lr, beta1, beta2, eps, weight_decay;
    static AdamParams get_default();
};

struct AdamVariable {
    float *data, *grad, *m, *v;
    int size;
public:
    AdamVariable(Variable*);
    ~AdamVariable();
};

class Adam {
    AdamParams params;
    int step_count;
    std::vector<AdamVariable> vars;
public:
    Adam();
    Adam(std::vector<Variable*> vars, AdamParams params);
    void step();
};

#define OPTIM_H
#endif