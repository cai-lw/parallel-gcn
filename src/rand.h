#ifndef RAND_H
#include <cstdlib>
#include <cstdint>
#include "immintrin.h"
#include <assert.h>


#ifdef OMP
#include <omp.h>
#endif

#define MY_RAND_MAX 0x7fffffff

void init_rand_state();
const int CACHE_LINE_SIZE = 64;
const int MAX_NUM_THREADS = 64;

uint32_t xorshift128plus(uint64_t* state);

#ifdef OMP
extern uint64_t rand_state[MAX_NUM_THREADS * CACHE_LINE_SIZE / sizeof(unsigned int)];
#define RAND() xorshift128plus(&rand_state[omp_get_thread_num() * CACHE_LINE_SIZE / sizeof(unsigned int)])
#else
extern uint64_t rand_state[2];
#define RAND() xorshift128plus(&rand_state[0])
#endif

__m256i gen_simd_rand(int thread_id);

#define RAND_H
#endif