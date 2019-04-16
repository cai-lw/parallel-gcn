#ifndef MODEL_H
#include <vector>
#include "variable.h"
#include "module.h"
#include "optim.h"

struct GCNParams {
    float num_nodes, input_dim, hidden_dim, output_dim, dropout, learning_rate, weight_decay, epochs, early_stopping;
    static GCNParams get_default();
};

class GCN {
    GCNParams params;
    std::vector<Module*> modules;
    std::vector<Variable> variables;
    Adam optimizer;
    float loss;
public:
    GCN(GCNParams);
    ~GCN();
    void train();
    void test();
};

#define MODEL_H
#endif