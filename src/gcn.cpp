#include "gcn.h"
#include "rand.h"
#include "timer.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <tuple>

GCNParams GCNParams::get_default() {
    return {2708, 1433, 16, 7, 0.5, 0.01, 5e-4, 100, 0};
}

GCN::GCN(GCNParams params, GCNData *input_data) {
    init_rand_state();
    this->params = params;
    data = input_data;
    modules.reserve(8);
    variables.reserve(8);
    variables.emplace_back(data->feature_index.indices.size(), false);
    input = &variables.back();
    modules.push_back(new Dropout(input, params.dropout));
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var1 = &variables.back();
    variables.emplace_back(params.input_dim * params.hidden_dim, true, true);
    Variable *layer1_weight = &variables.back();
    layer1_weight->glorot(params.input_dim, params.hidden_dim);
    // sparsematmul
    modules.push_back(new SparseMatmul(input, layer1_weight, layer1_var1, &data->feature_index, params.num_nodes, params.input_dim, params.hidden_dim));
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var2 = &variables.back();
    // graphsum
    modules.push_back(new GraphSum(layer1_var1, layer1_var2, &data->graph, params.hidden_dim));
    // RELU
    modules.push_back(new ReLU(layer1_var2));
    // dropout
    modules.push_back(new Dropout(layer1_var2, params.dropout));
    variables.emplace_back(params.num_nodes * params.output_dim);
    Variable *layer2_var1 = &variables.back();
    variables.emplace_back(params.hidden_dim * params.output_dim, true, true);
    Variable *layer2_weight = &variables.back();
    layer2_weight->glorot(params.hidden_dim, params.output_dim);
    // dense matrix multiply
    modules.push_back(new Matmul(layer1_var2, layer2_weight, layer2_var1, params.num_nodes, params.hidden_dim, params.output_dim));
    variables.emplace_back(params.num_nodes * params.output_dim);
    output = &variables.back();
    // graph sum
    modules.push_back(new GraphSum(layer2_var1, output, &data->graph, params.output_dim));
    truth = std::vector<int>(params.num_nodes);
    // cross entropy loss
    modules.push_back(new CrossEntropyLoss(output, truth.data(), &loss, params.output_dim));
    
    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params.learning_rate;
    adam_params.weight_decay = params.weight_decay;
    optimizer = Adam({{layer1_weight, true}, {layer2_weight, false}}, adam_params);
}

GCN::~GCN(){
    for(auto m: modules)
        delete m;
}

void GCN::set_input() {
    #ifdef SIMD
    #pragma omp parallel for simd schedule(static)
    #else
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < input->data.size(); i++)
        input->data[i] = data->feature_value[i];
}

void GCN::set_truth(int current_split) {
    #ifdef SIMD
    #pragma omp parallel for simd schedule(static)
    #else
    #pragma omp parallel for schedule(static)
    #endif
    for(int i = 0; i < params.num_nodes; i++)
        truth[i] = data->split[i] == current_split ? data->label[i] : -1;
}

float GCN::get_accuracy() {
    int wrong = 0, total = 0;
    #pragma omp parallel for schedule(static) reduction(+:total) reduction(+:wrong)
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

float GCN::get_l2_penalty() {
    float l2 = 0;
    #ifdef SIMD
    #pragma omp parallel for simd schedule(static) reduction(+:l2)
    #else
    #pragma omp parallel for schedule(static) reduction(+:l2)
    #endif
    for (int i = 0; i < variables[2].data.size(); i++) {
        float x = variables[2].data[i];
        l2 += x * x;
    }
    return params.weight_decay * l2 / 2;
}

std::pair<float, float> GCN::train_epoch() {
//    START_CLOCK(SET_INPUT);
    set_input();
//    END_CLOCK(SET_INPUT);

//    START_CLOCK(SET_TRUTH);
    set_truth(1);
//    END_CLOCK(SET_TRUTH);

//    START_CLOCK(FORWARD);
    for (auto m: modules)
        m->forward(true);
//    END_CLOCK(FORWARD);

//    START_CLOCK(Envaluate);
    float train_loss = loss + get_l2_penalty();
    float train_acc = get_accuracy();
//    END_CLOCK(Envaluate);

//    START_CLOCK(CALCULATE_Grad);
    for (int i = modules.size() - 1; i >= 0; i--)
        modules[i]->backward();
//    END_CLOCK(CALCULATE_Grad);

//    START_CLOCK(OPTIMIZE);
    optimizer.step();
//    END_CLOCK(OPTIMIZE);

    return {train_loss, train_acc};
}

std::pair<float, float> GCN::eval(int current_split) {
    set_input();
    set_truth(current_split);
    for (auto m: modules)
        m->forward(false);
    float test_loss = loss + get_l2_penalty();
    float test_acc = get_accuracy();
    return {test_loss, test_acc};
}

void GCN::run() {
    int epoch = 1;
    float total_time = 0.0;
    std::vector<float> loss_history;
    for(; epoch <= params.epochs; epoch++) {
        float train_loss, train_acc, val_loss, val_acc;
        timer_start(TMR_TRAIN);
        std::tie(train_loss, train_acc) = train_epoch();
        std::tie(val_loss, val_acc) = eval(2);
        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f time=%.5f\n",
            epoch, train_loss, train_acc, val_loss, val_acc, timer_stop(TMR_TRAIN));
        loss_history.push_back(val_loss);
        if(params.early_stopping > 0 && epoch >= params.early_stopping) {
            float recent_loss = 0.0;
            for(int i = epoch - params.early_stopping; i < epoch; i++)
                recent_loss += loss_history[i];
            if (val_loss > recent_loss / params.early_stopping) {
                printf("Early stopping...\n");
                break;
            }
        }
    }
    PRINT_TIMER_AVERAGE(TMR_TRAIN, epoch);
    PRINT_TIMER_AVERAGE(TMR_MATMUL_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_MATMUL_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_SPMATMUL_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_SPMATMUL_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_GRAPHSUM_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_GRAPHSUM_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_RELU_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_RELU_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_DROPOUT_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_DROPOUT_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_LOSS_FW, epoch);
    

    float test_loss, test_acc;
    timer_start(TMR_TEST);
    std::tie(test_loss, test_acc) = eval(3);
    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc, timer_stop(TMR_TEST));
}
