#ifndef SPARSE_H

/* For both sparse matrix and graph. Sparse matrix = sparse index + variable
 * Compressed Sparse Row (CSR) format: len(indices) = nnz, len(indptr) = nrow+1
 * indices[j] (j in [indptr[i], indptr[i+1])) stores column indices of nonzero
 * elements in the i-th row.
 */
struct SparseIndex {
    int *indices, *indptr, nrow, nnz;
};

#define SPARSE_H
#endif