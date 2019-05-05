#include <immintrin.h>
#include <iostream>
#include "rand.h"

#ifdef SIMD
#include "simdxorshift128plus.h"
avx_xorshift128plus_key_t mykey[MAX_NUM_THREADS * CACHE_LINE_SIZE / sizeof(avx_xorshift128plus_key_t)];
#endif

#ifdef OMP
#include "omp.h"
uint64_t rand_state[MAX_NUM_THREADS * CACHE_LINE_SIZE / sizeof(unsigned int)];
#else
uint64_t rand_state[2];
#endif









void init_rand_state() {
    #ifdef OMP
    for (int i = 0; i < MAX_NUM_THREADS; i++) {
        int x = 0, y = 0;
        while (x == 0 || y== 0) {
            x = rand();
            y = rand();
        }
        rand_state[i * CACHE_LINE_SIZE / sizeof(unsigned int)] = x;
        rand_state[i * CACHE_LINE_SIZE / sizeof(unsigned int) + 1] = y;
    }
#else
    int x = 0, y = 0;
        while (x == 0 || y== 0) {
            x = rand();
            y = rand();
        }
        rand_state[0] = x;
        rand_state[1] = y;
#endif

#ifdef SIMD
    for (int i = 0; i < MAX_NUM_THREADS; i++) {
        int x = 0, y = 0;
        while (x == 0 || y == 0) {
            x = rand();
            y = rand();
        }
        avx_xorshift128plus_init(x, y, &mykey[i * CACHE_LINE_SIZE / sizeof(avx_xorshift128plus_key_t)]);
    }
#endif
}

#ifdef SIMD
__m256i gen_simd_rand(int thread_id) {
//    std::cout << "omp_get_thread_num() = " << thread_id << std::endl;
    return avx_xorshift128plus(&mykey[thread_id * CACHE_LINE_SIZE / sizeof(avx_xorshift128plus_key_t)]);
};
#endif

uint32_t xorshift128plus(uint64_t* state) {
    uint64_t t = state[0];
    uint64_t const s = state[1];
    assert(t && s);
    state[0] = s;
    t ^= t << 23;		// a
    t ^= t >> 17;		// b
    t ^= s ^ (s >> 26);	// c
    state[1] = t;
    uint32_t res = (t + s) & 0x7fffffff;
    return res;
}
