#ifndef OPTIM_H
#include <vector>
#include <utility>
#include "variable.h"

struct AdamParams {
    float lr, beta1, beta2, eps, weight_decay;
    static AdamParams get_default();
};

struct AdamVariable {
    std::vector<float> *data, *grad, m, v;
    bool decay;
public:
    int size();
    AdamVariable(Variable*, bool);
};

class Adam {
    AdamParams params;
    int step_count;
    std::vector<AdamVariable> vars;
public:
    Adam() {}
    Adam(std::vector<std::pair<Variable*, bool>> vars, AdamParams params);
    void step();
};

#define OPTIM_H
#endif