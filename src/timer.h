#ifndef TIMER_H
#include <chrono>

#define START_CLOCK(X) auto __timer_t0_##X = std::chrono::high_resolution_clock::now();
#define GET_CLOCK(X) std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - __timer_t0_##X).count()


#ifdef debug
#define PRINT_CLOCK(X) fprintf(stderr, #X " took %f ms\n", GET_CLOCK(X) * 1000);
#else
#define PRINT_CLOCK(X)
#endif

class Timer
{
public:
    Timer(const std::string& name)
            : name_ (name),
              start_ (std::clock())
    {
    }
    ~Timer()
    {
#ifdef debug
        double elapsed = (double(std::clock() - start_) / double(CLOCKS_PER_SEC));
        std::cout << name_ << ": " << int(elapsed * 1000) << "ms" << std::endl;
#endif
    }
private:
    std::string name_;
    std::clock_t start_;
};




#define TIMER_H
#endif