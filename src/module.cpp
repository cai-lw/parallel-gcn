#include "module.h"
#include "rand.h"
#include <cstdlib>
#include <cmath>

#ifdef OMP
#include <omp.h>
#endif

Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) :
        a(a), b(b), c(c), m(m), n(n), p(p) {}

void Matmul::forward(bool training) {
    c->zero();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
#ifdef SIMD
#pragma omp simd
#endif
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
        }
}

void Matmul::backward() {
    a->zero_grad();
    b->zero_grad();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
#ifdef SIMD
#pragma omp simd
#endif
                for (int k = 0; k < p; k++) {
                    a->grad[i * n + j] += c->grad[i * p + k] * b->data[j * p + k];
#ifdef OMP
                    b->local_grad[omp_get_thread_num()][j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
#else
                    b->grad[j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
#endif
                }
#ifdef OMP
#pragma omp parallel for
    for(int i = 0; i < b->grad.size(); i++)
        for(int thread = 0; thread < omp_get_num_threads(); thread++)
            b->grad[i] += b->local_grad[thread][i];
#endif
}

SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p) :
        a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

void SparseMatmul::forward(bool training) {
    c->zero();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
#ifdef SIMD
#pragma omp simd
#endif
            for (int k = 0; k < p; k++)
                c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
        }
}

void SparseMatmul::backward() {
    b->zero_grad();
    int row = 0;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < sp->indptr.size() - 1; i++)
        for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
            int j = sp->indices[jj];
#ifdef SIMD
#pragma omp simd
#endif
            for (int k = 0; k < p; k++)
#ifdef OMP
                b->local_grad[omp_get_thread_num()][j * p + k] += c->grad[i * p + k] * a->data[jj];
#else
                    b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
#endif
        }
#ifdef OMP
#pragma omp parallel for
    for(int i = 0; i < b->grad.size(); i++)
        for(int thread = 0; thread < omp_get_num_threads(); thread++)
            b->grad[i] += b->local_grad[thread][i];
#endif
}

GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim) :
        in(in), out(out), graph(graph), dim(dim) {}

void GraphSum::forward(bool training) {
    out->zero();
#pragma omp parallel for schedule(static)
    for (int src = 0; src < graph->indptr.size() - 1; src++)
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                    (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );
#ifdef SIMD
#pragma omp simd
#endif
            for (int j = 0; j < dim; j++)
                // This only works for undirected graphs. Should be out[dst] += coef * in[src]
                out->data[src * dim + j] += coef * in->data[dst * dim + j];
        }
}

void GraphSum::backward() {
    in->zero_grad();
#pragma omp parallel for schedule(static)
    for (int src = 0; src < graph->indptr.size() - 1; src++)
        for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
            int dst = graph->indices[i];
            float coef = 1.0 / sqrtf(
                    (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
            );
#ifdef SIMD
#pragma omp simd
#endif
            for (int j = 0; j < dim; j++)
                in->grad[src * dim + j] += coef * out->grad[dst * dim + j];
        }
}

CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) :
        logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

void CrossEntropyLoss::forward(bool training) {
    float total_loss = 0;
    int count = 0;
    if (training) logits->zero_grad();
#pragma omp parallel for schedule(static) reduction(+:total_loss) reduction(+:count)
    for (int i = 0; i < logits->data.size() / num_classes; i++) {
        if (truth[i] < 0) continue;
        count++;
        float *logit = &logits->data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;
#ifdef SIMD
#pragma omp simd
#endif
        for (int j = 0; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);
#ifdef SIMD
#pragma omp simd
#endif
        for (int j = 0; j < num_classes; j++) {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }
        total_loss += logf(sum_exp) - logit[truth[i]];

        if (training) {
#ifdef SIMD
#pragma omp simd
#endif
            for (int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
            logits->grad[i * num_classes + truth[i]] -= 1.0;
        }
    }
    *loss = total_loss / count;
    if (training)
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < logits->grad.size(); i++)
            logits->grad[i] /= count;
}

void CrossEntropyLoss::backward() {
}

ReLU::ReLU(Variable *in) {
    this->in = in;
    mask = new bool[in->data.size()];
}

ReLU::~ReLU() {
    delete[] mask;
}

void ReLU::forward(bool training) {
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = in->data[i] > 0;
        if (training) mask[i] = keep;
        if (!keep) in->data[i] = 0;
    }
}

void ReLU::backward() {
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < in->data.size(); i++)
        if (!mask[i]) in->grad[i] = 0;
}

Dropout::Dropout(Variable *in, float p) {
    this->in = in;
    this->p = p;
    mask = new bool[in->data.size()];
}

Dropout::~Dropout() {
    delete[] mask;
}

void Dropout::forward(bool training) {
    if (!training) return;
    const int threshold = int(p * RAND_MAX);
    float scale = 1 / (1 - p);
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < in->data.size(); i++) {
        bool keep = RAND() >= threshold;
        mask[i] = keep;
        in->data[i] *= mask[i] ? scale : 0;
    }
}

void Dropout::backward() {
    float scale = 1 / (1 - p);
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < in->data.size(); i++)
        in->grad[i] *= mask[i] ? scale : 0;
}
