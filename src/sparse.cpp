//
// Created by Chengze Fan on 2019-04-18.
//
#include "sparse.h"
#include <cstdio>

using namespace std;

void SparseIndex::print() {
    for(int i = 0; i < rows.size(); i++)
        printf("%d-%d ", rows[i], cols[i]);
    printf("\n");
}