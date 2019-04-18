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

class GCNData {
public:
    SparseIndex *feature_index, *graph;
    std::vector<int> split;
    std::vector<int> label;
    std::vector<float> feature_value;

    GCNData();
};

class GCN {
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
    GCNData data;
public:
    GCN(GCNParams params, GCNData data);
    GCN();
    GCNParams params;
    ~GCN();
    void run();
};

#define GCN_H
#endif