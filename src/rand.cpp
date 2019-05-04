#include <immintrin.h>
#include <iostream>
#include "rand.h"

#ifdef SIMD
#include "simdxorshift128plus.h"
#endif

#ifdef OMP
#include "omp.h"
unsigned int rand_state[MAX_NUM_THREADS * PADDING_FOR_CACHE_LINE];
#endif

avx_xorshift128plus_key_t mykey[MAX_NUM_THREADS];
__m256i MAX_V = _mm256_set1_epi32(RAND_MAX);

void init_rand_state() {
#ifdef OMP
    for (int i = 0; i < MAX_NUM_THREADS; i++) {
        rand_state[i * PADDING_FOR_CACHE_LINE] = rand();
    }
#endif
#ifdef SIMD
    for (int i = 0; i < MAX_NUM_THREADS; i++) {
        int x = 1, y = 1;
        while (x == 0 || y == 0) {
            x = rand();
            y = rand();
        }
        avx_xorshift128plus_init(x, y, &mykey[i]);
    }
#endif
}
#ifdef SIMD
__m256i gen_simd_rand() {
    return avx_randombound_epu32(avx_xorshift128plus(&mykey[omp_get_thread_num()]), MAX_V) ;
};
#endif

#ifdef OMP
uint32_t xorshift32(uint32_t* state) {
    uint32_t x = state[0];
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state[0] = x;
}
#endif
