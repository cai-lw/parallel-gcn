#include <chrono>

#define START_CLOCK(X) auto __timer_t0_##X = std::chrono::high_resolution_clock::now();
#define GET_CLOCK(X) std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - __timer_t0_##X).count()

#ifdef debug
#define PRINT_CLOCK(X) fprintf(stderr, #X " took %f ms\n", GET_CLOCK(X) * 1000);
#else
#define PRINT_CLOCK(X)
#endif