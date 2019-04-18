//
// Created by Chengze Fan on 2019-04-18.
//
#include "sparse.h"

using namespace std;

void SparseIndex::print() {
    std::cout << "---sparse index info--" << endl;

    std::cout << "indptr: ";
    for (auto i: indptr) {
        cout << i << " ";
    }
    cout << endl;

    cout << "indices: ";
    for (auto i: indices) {
        cout << i << " ";
    }
    cout << endl;
}