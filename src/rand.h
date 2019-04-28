#ifndef RAND_H
#include <cstdlib>
#ifdef OMP
#include <omp.h>
#endif 

void init_rand_state();

#ifdef OMP
// Cache line size is 64 bytes for x86. https://stackoverflow.com/questions/7281699/aligning-to-cache-line-and-knowing-the-cache-line-size/7284876
const int PADDING_FOR_CACHE_LINE = 64 / sizeof(unsigned int);
const int MAX_NUM_THREADS = 64;
extern unsigned int rand_state[MAX_NUM_THREADS * PADDING_FOR_CACHE_LINE];
#define RAND() rand_r(&rand_state[omp_get_thread_num() * PADDING_FOR_CACHE_LINE])
#else
#define RAND() rand()
#endif

#define RAND_H
#endif