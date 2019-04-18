//
// Created by Chengze Fan on 2019-04-17.
//

#include <sstream>
#include "parser.h"
using namespace std;
Parser::Parser(GCNParams *gcnParams, GCNData *gcnData, std::string graph_name) {
    this->graph_file.open(graph_name + ".graph");
    this->split_file.open(graph_name + ".split");
    this->svmlight_file.open(graph_name + ".svmlight");
    this->gcnParams = gcnParams;
    this->gcnData = gcnData;
}

void Parser::parseGraph() {
    auto &graph_sparse_index = this->gcnData->graph;

    graph_sparse_index->indptr.push_back(0);
    for (int node = 0; !graph_file.eof(); node++) {
        graph_sparse_index->indptr.push_back(graph_sparse_index->indptr.back());
        std::string line;
        getline(graph_file, line);
        std::istringstream ss(line);
        while (!line.empty() && !ss.eof()) {
            int neighbor;
            ss >> neighbor;
            graph_sparse_index->indices.push_back(neighbor);
            graph_sparse_index->indptr.back() += 1;
        }
    }
    graph_sparse_index->nnz = graph_sparse_index->indices.size();
    graph_sparse_index->nrow = graph_sparse_index->indptr.size() - 1;
}

bool Parser::isValidInput() {
    return graph_file.is_open() && split_file.is_open() && svmlight_file.is_open();
}

void Parser::parseNode() {
    auto &feature_sparse_index = this->gcnData->feature_index;
    auto &feature_val = this->gcnData->feature_value;
    auto &labels = this->gcnData->label;

    feature_sparse_index->indptr.push_back(0);

    for (int node = 0; !svmlight_file.eof(); node++) {
        feature_sparse_index->indptr.push_back(feature_sparse_index->indptr.back());

        std::string line;
        getline(svmlight_file, line);
        std::istringstream ss(line);

        int label;
        ss >> label;
        labels.push_back(label);

        while (!line.empty() && !ss.eof()) {
            string kv;
            ss >> kv;
            std::istringstream kv_ss(kv);

            int k;
            float v;
            char col;
            kv_ss >> k >> col >> v;

            feature_val.push_back(v);
            feature_sparse_index->indices.push_back(k);
            feature_sparse_index->indptr.back() += 1;
        }
    }
    feature_sparse_index->nnz = feature_sparse_index->indices.size();
    feature_sparse_index->nrow = feature_sparse_index->indptr.size() - 1;
}

void Parser::parseSplit() {
    auto &split = this->gcnData->split;

    while (!split_file.eof()) {
        int s;
        split_file >> s;
        split.push_back(s);
    }
}

bool Parser::parse() {
    if (!isValidInput()) return false;
    this->parseGraph();
    this->parseNode();
    this->parseSplit();
    return true;
}

