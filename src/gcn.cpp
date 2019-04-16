#include "gcn.h"

GCNParams GCNParams::get_default() {
    return {2708, 1433, 16, 7, 0.5, 0.01, 5e-4, 200, 10};
}

GCN::GCN(GCNParams params) {
    modules.push_back(new Dropout());
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var1 = &variables.back();
    variables.emplace_back(params.input_dim * params.hidden_dim);
    Variable *layer1_weight = &variables.back();
    layer1_weight->glorot(params.input_dim, params.hidden_dim);
    modules.push_back(new SparseMatmul());
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var2 = &variables.back();
    modules.push_back(new GraphSum());
    modules.push_back(new ReLU(layer1_var2));
    modules.push_back(new Dropout(layer1_var2, params.dropout));
    variables.emplace_back(params.num_nodes * params.output_dim);
    Variable *layer2_var1 = &variables.back();
    variables.emplace_back(params.hidden_dim * params.output_dim);
    Variable *layer2_weight = &variables.back();
    layer2_weight->glorot(params.hidden_dim, params.output_dim);
    modules.push_back(new Matmul(layer1_var2, layer2_weight));
    variables.emplace_back(params.num_nodes * params.output_dim);
    Variable *layer2_var2 = &variables.back();
    modules.push_back(new GraphSum());
    modules.push_back(new CrossEntropyLoss());

    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params.learning_rate;
    adam_params.weight_decay = params.weight_decay;
    optimizer = Adam({&layer1_weight, &layer2_weight}, adam_params);
}

GCN::~GCN(){
    for (auto m: modules)
        delete m;
}

void GCN::train() {
    // TODO: Implement the training logic
    for(int epoch = 0; epoch < params.epochs; epoch++) {
        for (auto m: modules)
            m->forward(true);
        for (int i = modules.size() - 1; i >= 0; i--)
            modules[i]->backward();
        optimizer.step();
    }
}

void GCN::test() {
    // TODO: Implement the testing logic
    for (auto m: modules)
        m->forward(false);
}