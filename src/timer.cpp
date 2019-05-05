#include "timer.h"

std::chrono::time_point<std::chrono::high_resolution_clock> tmr_t0[__NUM_TMR];
float tmr_sum[__NUM_TMR];
float tmr_count[__NUM_TMR];

void timer_start(timer_t t) {
    tmr_t0[t] = std::chrono::high_resolution_clock::now();
}

float timer_stop(timer_t t) {
    float count = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - tmr_t0[t]).count();
    tmr_sum[t] += count;
    tmr_count[t]++;
    return count;
}

float timer_average(timer_t t) {
    return tmr_sum[t] / tmr_count[t];
}