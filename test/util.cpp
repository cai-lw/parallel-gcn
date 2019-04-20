#include "util.h"
#include <cmath>

bool allclose(std::vector<float> x, std::vector<float> y, float eps) {
    for(int i = 0; i < x.size(); i++)
        if(abs(x[i] - y[i]) > eps) return false;
    return true;
}