#ifndef TIMER_H
#include <chrono>
#include <vector>

typedef enum {
    TMR_TRAIN = 0,
    TMR_TEST,
    TMR_MATMUL_FW,
    TMR_MATMUL_BW,
    TMR_SPMATMUL_FW,
    TMR_SPMATMUL_BW,
    TMR_GRAPHSUM_FW,
    TMR_GRAPHSUM_BW,
    TMR_LOSS_FW,
    TMR_RELU_FW,
    TMR_RELU_BW,
    TMR_DROPOUT_FW,
    TMR_DROPOUT_BW,
    __NUM_TMR
} timer_instance;

void timer_start(timer_instance t);
float timer_stop(timer_instance t);
float timer_total(timer_instance t);

#define PRINT_TIMER_AVERAGE(T, E) printf(#T " average time: %.3fms\n", timer_total(T) * 1000 / E)

#define TIMER_H
#endif
