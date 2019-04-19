#ifndef SPARSE_H
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

/* For both sparse matrix and graph. Sparse matrix = sparse index + variable
 * Compressed Sparse Row (CSR) format: len(indices) = nnz, len(indptr) = nrow+1
 * indices[j] (j in [indptr[i], indptr[i+1])) stores column indices of nonzero
 * elements in the i-th row.
 */
class SparseIndex {
public:
    std::vector<int> indices;
    std::vector<int> indptr;
    void print();
};


#define SPARSE_H
#endif