#include "rand.h"

#ifdef OMP
unsigned int rand_state[MAX_NUM_THREADS * PADDING_FOR_CACHE_LINE];
#endif

void init_rand_state() {
    #ifdef OMP
    for(int i = 0; i < MAX_NUM_THREADS; i++)
        rand_state[i * PADDING_FOR_CACHE_LINE] = rand();
    #endif
}