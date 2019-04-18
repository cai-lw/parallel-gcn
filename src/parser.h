//
// Created by Chengze Fan on 2019-04-17.
//

#ifndef PARALLEL_GCN_PARSER_H
#define PARALLEL_GCN_PARSER_H

#include "sparse.h"
#include "gcn.h"
#include <string>
#include <iostream>
#include <fstream>

class Parser {
public:
    Parser(GCNParams *gcnParams, GCNData* gcnData, std::string graph_name);
    bool parse();
private:
    std::ifstream graph_file;
    std::ifstream split_file;
    std::ifstream svmlight_file;
    GCNParams *gcnParams;
    GCNData *gcnData;
    void parseGraph();
    void parseNode();
    void parseSplit();
    bool isValidInput();
};


#endif //PARALLEL_GCN_PARSER_H
