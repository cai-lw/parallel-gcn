//
// Created by Chengze Fan on 2019-04-17.
//

#include <sstream>
#include <algorithm>
#include <cmath>
#include "parser.h"

using namespace std;
Parser::Parser(GCNParams *gcnParams, GCNData *gcnData, std::string graph_name) {
    this->graph_file.open("data/" + graph_name + ".graph");
    this->split_file.open("data/" + graph_name + ".split");
    this->svmlight_file.open("data/" + graph_name + ".svmlight");
    this->gcnParams = gcnParams;
    this->gcnData = gcnData;
}

void Parser::parseGraph() {
    auto &graph_sparse_index = this->gcnData->graph;

    graph_sparse_index.indptr.push_back(0);
    int node = 0;
    while(true) {
        std::string line;
        getline(graph_file, line);
        if (graph_file.eof()) break;
        
        // Implicit self connection
        graph_sparse_index.indices.push_back(node);
        graph_sparse_index.indptr.push_back(graph_sparse_index.indptr.back() + 1);
        node++;

        std::istringstream ss(line);
        while (true) {
            int neighbor;
            ss >> neighbor;
            if (ss.fail()) break;
            graph_sparse_index.indices.push_back(neighbor);
            graph_sparse_index.indptr.back() += 1;
        }
    }
    
    gcnParams->num_nodes = node;
}

bool Parser::isValidInput() {
    return graph_file.is_open() && split_file.is_open() && svmlight_file.is_open();
}

void Parser::parseNode() {
    auto &feature_sparse_index = this->gcnData->feature_index;
    auto &feature_val = this->gcnData->feature_value;
    auto &labels = this->gcnData->label;

    feature_sparse_index.indptr.push_back(0);

    int max_idx = 0, max_label = 0;
    while(true) {
        std::string line;
        getline(svmlight_file, line);
        if (svmlight_file.eof()) break;
        feature_sparse_index.indptr.push_back(feature_sparse_index.indptr.back());
        std::istringstream ss(line);

        int label = -1;
        ss >> label;
        labels.push_back(label);
        if (ss.fail()) continue;
        max_label = max(max_label, label);

        while (true) {
            string kv;
            ss >> kv;
            if(ss.fail()) break;
            std::istringstream kv_ss(kv);

            int k;
            float v;
            char col;
            kv_ss >> k >> col >> v;

            feature_val.push_back(v);
            feature_sparse_index.indices.push_back(k);
            feature_sparse_index.indptr.back() += 1;
            max_idx = max(max_idx, k);
        }
    }
    gcnParams->input_dim = max_idx + 1;
    gcnParams->output_dim = max_label + 1;
}

void Parser::parseSplit() {
    auto &split = this->gcnData->split;

    while (true) {
        std::string line;
        getline(split_file, line);
        if (split_file.eof()) break;
        split.push_back(std::stoi(line));
    }
}

void vprint(std::vector<int> v){
    for(int i:v)printf("%i ", i);
    printf("\n");
}

bool Parser::parse() {
    if (!isValidInput()) return false;
    this->parseGraph();
    std::cout << "Parse Graph Succeeded." << endl;
    this->parseNode();
    std::cout << "Parse Node Succeeded." << endl;
    this->parseSplit();
    std::cout << "Parse Split Succeeded." << endl;
    return true;
}

