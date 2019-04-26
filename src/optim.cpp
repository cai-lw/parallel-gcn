#include "optim.h"
#include <cmath>
#include <cstdlib>

AdamParams AdamParams::get_default() {
    return {0.001, 0.9, 0.999, 1e-8, 0.0};
}

AdamVariable::AdamVariable(Variable *var, bool decay):
    data(&var->data), grad(&var->grad), m(var->data.size(), 0.0), v(var->data.size(), 0.0), decay(decay) {}

int AdamVariable::size() {
    return data->size();
}

Adam::Adam(std::vector<std::pair<Variable*, bool>> vars, AdamParams params){
    step_count = 0;
    this->params = params;
    for (auto v: vars)
        this->vars.emplace_back(v.first, v.second);
}

void Adam::step(){
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) / (1 - powf(params.beta1, step_count));
    for (auto &var: vars) {
        for (int i = 0; i < var.size(); i++) {
            float grad = (*var.grad)[i];
            if (var.decay) grad += params.weight_decay * (*var.data)[i];
            var.m[i] = params.beta1 * var.m[i] + (1.0 - params.beta1) * grad;
            var.v[i] = params.beta2 * var.v[i] + (1.0 - params.beta2) * grad * grad;
            (*var.data)[i] -= step_size * var.m[i] / (sqrtf(var.v[i]) + params.eps);
        }
    }
}
