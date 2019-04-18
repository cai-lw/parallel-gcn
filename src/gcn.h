#ifndef GCN_H
#include <vector>
#include <utility>
#include "variable.h"
#include "sparse.h"
#include "module.h"
#include "optim.h"

struct GCNParams {
    int num_nodes, input_dim, hidden_dim, output_dim;
    float dropout, learning_rate, weight_decay;
    int epochs, early_stopping;
    static GCNParams get_default();
};

struct GCNData {
    float *feature_value;
    SparseIndex *feature_index, *graph;
    int *split, *label;
};

class GCN {
    GCNParams params;
    GCNData data;
    std::vector<Module*> modules;
    std::vector<Variable> variables;
    Variable *input, *output;
    int *truth;
    Adam optimizer;
    float loss;
    std::vector<float> loss_history;
    void set_input(bool training);
    void set_truth(int current_split);
    float get_accuracy();
    std::pair<float, float> train_epoch();
    std::pair<float, float> eval(int current_split);
public:
    GCN(GCNParams params, GCNData data);
    ~GCN();
    void run();
};

#define GCN_H
#endif