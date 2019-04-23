#include <catch2/catch.hpp>
#include "../src/optim.h"
#include "util.h"

TEST_CASE("Adam optimizer") {
    Variable a(4);
    a.data = {0.4027, 0.2111, 0.4161, 0.9383};
    std::vector<float> a100_ref = {0.3073, 0.1205, 0.3205, 0.8402};
    AdamParams params = AdamParams::get_default();
    Adam adam({{&a, false}}, params);
    for(int i = 0; i < 100; i++) {
        a.grad = a.data;
        adam.step();
    }
    REQUIRE(allclose(a.data, a100_ref));
}
