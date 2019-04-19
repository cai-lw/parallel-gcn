#include "module.h"
#include <cstdlib>
#include <cmath>

Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p):
    a(a), b(b), c(c), m(m), n(n), p(p) {}

void Matmul::forward(bool training) {
    c->zero();
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
}

void Matmul::backward() {
    a->zero_grad();
    b->zero_grad();
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < p; k++) {
                a->grad[i * n + j] += c->grad[i * p + k] * b->data[j * p + k];
                b->grad[j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
            }
}

SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p):
    a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

void SparseMatmul::forward(bool training) {
    c->zero();
    int row = 0;
    for(int i = 0; i < sp->indices.size(); i++) {
        while(i >= sp->indptr[row + 1]) row++;
        int col = sp->indices[i];
        for(int k = 0; k < p; k++) {
            c->data[row * p + k] += a->data[i] * b->data[col * p + k];
        }
    }
}

void SparseMatmul::backward() {
    b->zero_grad();
    int row = 0;
    for(int i = 0; i < sp->indices.size(); i++) {
        while(i >= sp->indptr[row + 1]) row++;
        int col = sp->indices[i];
        for(int k = 0; k < p; k++) {
            b->grad[col * p + k] = c->grad[row * p + k] * a->data[i];
        }
    }
}

GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim):
    in(in), out(out), graph(graph), dim(dim) {}

void GraphSum::forward(bool training) {
    out->zero();
    int src = 0;
    for(int i = 0; i < graph->indices.size(); i++) {
        while(i >= graph->indptr[src + 1]) src++;
        int dst = graph->indices[i];
        float coef = 1.0 / sqrtf(
            (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
        );
        for(int j = 0; j < dim; j++) {
            out->data[dst * dim + j] += coef * in->data[src * dim + j];
        }
    }
}

void GraphSum::backward() {
    in->zero_grad();
    int src = 0;
    for(int i = 0; i < graph->indices.size(); i++) {
        while(i >= graph->indptr[src + 1]) src++;
        int dst = graph->indices[i];
        float coef = 1.0 / sqrtf(
            (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
        );
        for(int j = 0; j < dim; j++) {
            in->grad[src * dim + j] += coef * out->grad[dst * dim + j];
        }
    }
}

CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes):
    logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

void CrossEntropyLoss::forward(bool training) {
    *loss = 0;
    int count = 0;
    if(training) logits->zero_grad();
    for(int i = 0; i < logits->size() / num_classes; i++) {
        if (truth[i] < 0) continue;
        count++;
        float* logit = &logits->data[i * num_classes];
        float sum_exp = 0.0;
        for(int j = 0; j < num_classes; j++)
            sum_exp += expf(logit[j]);
        *loss += logf(sum_exp) - logit[truth[i]];

        if(training) {
            for(int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
            logits->grad[i * num_classes + truth[i]] -= 1.0;
        }
    }
    *loss /= count;
    if(training)
        for(auto &g: logits->grad)
            g /= count;
}

void CrossEntropyLoss::backward() {
}

ReLU::ReLU(Variable *in) {
    this->in = in;
    mask = new bool[in->size()];
}

ReLU::~ReLU(){
    delete[] mask;
}

void ReLU::forward(bool training) {
    for (int i = 0; i < in->size(); i++) {
        bool keep = in->data[i] > 0;
        if (training) mask[i] = keep;
        if (!keep) in->data[i] = 0;
    }
}

void ReLU::backward() {
    for (int i = 0; i < in->size(); i++)
        if (!mask[i]) in->grad[i] = 0;
}

Dropout::Dropout(Variable *in, float p) {
    this->in = in;
    this->p = p;
    mask = new bool[in->size()];
}

Dropout::~Dropout(){
    delete[] mask;
}

void Dropout::forward(bool training) {
    if (!training) return;
    const int threshold = int(p * RAND_MAX);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->size(); i++) {
        bool keep = rand() >= threshold;
        mask[i] = keep;
        in->data[i] *= mask[i] ? scale : 0;
    }
}

void Dropout::backward() {
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
}
