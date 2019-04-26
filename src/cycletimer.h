#ifndef CYCLETIMER_H
/* Cycle timer code, adapted from CycleTimer.h found in 15-418 code repositories */

double currentSeconds();

#ifdef DEBUG
#define START_CLOCK(X) double X = currentSeconds() * 1000;
#define END_CLOCK(X) fprintf(stderr, #X " took %f ms\n", currentSeconds() * 1000 - X);
#else
#define START_CLOCK(X)
#define END_CLOCK(X)
#endif

#define CYCLETIMER_H
#endif
