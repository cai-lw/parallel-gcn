#ifndef MODULE_H

#include <immintrin.h>
#include "variable.h"
#include "sparse.h"

class Module {
public:
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual ~Module() {};
};

class Matmul: public Module {
    Variable *a, *b, *c;
    int m, n, p;
public:
    Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p);
    ~Matmul() {}
    void forward(bool);
    void backward();
};

class SparseMatmul: public Module {
    Variable *a, *b, *c;
    SparseIndex *sp;
    int m, n, p;
public:
    SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p);
    ~SparseMatmul() {}
    void forward(bool);
    void backward();
};

class GraphSum: public Module {
    Variable *in, *out;
    SparseIndex *graph;
    int dim;
public:
    GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim);
    ~GraphSum() {}
    void forward(bool);
    void backward();
};

class CrossEntropyLoss: public Module {
    Variable *logits;
    int *truth;
    float *loss;
    int num_classes;
public:
    CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes);
    ~CrossEntropyLoss() {}
    void forward(bool);
    void backward();
};

class ReLU: public Module {
    Variable *in;
    bool *mask;
public:
    ReLU(Variable *in);
    ~ReLU();
    void forward(bool);
    void backward();
};

class Dropout: public Module {
    Variable *in;
    int *mask;
    float p;
public:
    Dropout(Variable *in, float p);
    ~Dropout();
    void forward(bool);
    void backward();
};


#define MODULE_H
#endif