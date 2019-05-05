#include "timer.h"

std::chrono::time_point<std::chrono::high_resolution_clock> tmr_t0[__NUM_TMR];
float tmr_sum[__NUM_TMR];

void timer_start(timer_instance t) {
    tmr_t0[t] = std::chrono::high_resolution_clock::now();
}

float timer_stop(timer_instance t) {
    float count = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - tmr_t0[t]).count();
    tmr_sum[t] += count;
    return count;
}

float timer_total(timer_instance t) {
    return tmr_sum[t];
}
