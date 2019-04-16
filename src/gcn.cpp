#include "gcn.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <tuple>
#include <chrono>

GCNParams GCNParams::get_default() {
    return {2708, 1433, 16, 7, 0.5, 0.01, 5e-4, 200, 10};
}

GCN::GCN(GCNParams params, GCNData data) {
    this->params = params;
    this->data = data;
    variables.emplace_back(data.feature_index->nnz, false);
    input = &variables.back();
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var1 = &variables.back();
    variables.emplace_back(params.input_dim * params.hidden_dim);
    Variable *layer1_weight = &variables.back();
    layer1_weight->glorot(params.input_dim, params.hidden_dim);
    modules.push_back(new SparseMatmul(input, layer1_weight, layer1_var1, data.feature_index, params.num_nodes, params.input_dim, params.hidden_dim));
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var2 = &variables.back();
    modules.push_back(new GraphSum(layer1_var1, layer1_var2, data.graph, params.hidden_dim));
    modules.push_back(new ReLU(layer1_var2));
    modules.push_back(new Dropout(layer1_var2, params.dropout));
    variables.emplace_back(params.num_nodes * params.output_dim);
    Variable *layer2_var1 = &variables.back();
    variables.emplace_back(params.hidden_dim * params.output_dim);
    Variable *layer2_weight = &variables.back();
    layer2_weight->glorot(params.hidden_dim, params.output_dim);
    modules.push_back(new Matmul(layer1_var2, layer2_weight, layer2_var1, params.num_nodes, params.hidden_dim, params.output_dim));
    variables.emplace_back(params.num_nodes * params.output_dim);
    output = &variables.back();
    modules.push_back(new GraphSum(layer2_var1, output, data.graph, params.output_dim));
    truth = new int[params.num_nodes];
    modules.push_back(new CrossEntropyLoss(output, truth, &loss, params.output_dim));

    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params.learning_rate;
    adam_params.weight_decay = params.weight_decay;
    optimizer = Adam({&layer1_weight, &layer2_weight}, adam_params);
}

GCN::~GCN(){
    delete[] truth;
    for(auto m: modules)
        delete m;
}

void GCN::set_input(bool training) {
    if (!training) {
        memcpy(input->data, data.feature_value, input->size * sizeof(float));
        return;
    }
    const int threshold = int(params.dropout * RAND_MAX);
    float scale = 1 / (1 - params.dropout);
    for (int i = 0; i < input->size; i++) {
        bool drop = rand() < threshold;
        input->data[i] = drop ? 0 : data.feature_value[i] * scale;
    }
}

void GCN::set_truth(int current_split) {
    for(int i = 0; i < params.num_nodes; i++)
        truth[i] = data.split[i] == current_split ? data.label[i] : -1;
}

float GCN::get_accuracy() {
    int wrong = 0, total = 0;
    for(int i = 0; i < params.num_nodes; i++) {
        if(truth[i] < 0) continue;
        total++;
        float truth_logit = output->data[i * params.output_dim + truth[i]];
        for(int j = 0; j < params.output_dim; j++)
            if (output->data[i * params.output_dim + j] > truth_logit) {
                wrong++;
                break;
            }
    }
    return float(total - wrong) / total;
}

std::pair<float, float> GCN::train_epoch() {
    set_input(true);
    set_truth(1);
    for (auto m: modules)
        m->forward(true);
    float train_loss = loss;
    float train_acc = get_accuracy();
    for (int i = modules.size() - 1; i >= 0; i--)
        modules[i]->backward();
    optimizer.step();
    return {train_loss, train_acc};
}

std::pair<float, float> GCN::eval(int current_split) {
    set_input(false);
    set_truth(current_split);
    for (auto m: modules)
        m->forward(false);
    float test_loss = loss;
    float test_acc = get_accuracy();
    return {test_loss, test_acc};
}

void GCN::run() {
    for(int epoch = 0; epoch < params.epochs; epoch++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        float train_loss, train_acc, dev_loss, dev_acc;
        std::tie(train_loss, train_acc) = train_epoch();
        std::tie(dev_loss, dev_acc) = eval(2);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = t2 - t1;
        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f time=%.5f\n",
            epoch + 1, train_loss, train_acc, dev_loss, dev_acc, elapsed.count());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    float test_loss, test_acc;
    std::tie(test_loss, test_acc) = eval(3);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = t2 - t1;
    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc, elapsed.count());
}

