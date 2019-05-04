#ifndef RAND_H
#include <cstdlib>
#include "immintrin.h"

#ifdef OMP
#include <omp.h>
#endif 

void init_rand_state();

#ifdef OMP
// Cache line size is 64 bytes for x86. https://stackoverflow.com/questions/7281699/aligning-to-cache-line-and-knowing-the-cache-line-size/7284876
const int PADDING_FOR_CACHE_LINE = 64 / sizeof(unsigned int);
const int MAX_NUM_THREADS = 64;
extern unsigned int rand_state[MAX_NUM_THREADS * PADDING_FOR_CACHE_LINE];

uint32_t xorshift32(uint32_t* state);

//#define RAND() rand_r(&rand_state[omp_get_thread_num() * PADDING_FOR_CACHE_LINE])
#define RAND() xorshift32(&rand_state[omp_get_thread_num() * PADDING_FOR_CACHE_LINE])
#else
#define RAND() rand()
#endif

__m256i gen_simd_rand();

#define RAND_H
#endif