#include <catch2/catch.hpp>
#include "../src/module.h"
#include "util.h"

TEST_CASE("Matrix multiplication") {
    Variable a(6), b(6), c(4);
    a.data = {0.2211, 0.1726, 0.1673, 0.4715, 0.0824, 0.5858};
    b.data = {0.3101, 0.5706, 0.1343, 0.9268, 0.8920, 0.5085};
    std::vector<float> c_ref = {0.2410, 0.3712, 0.6798, 0.6433};
    std::vector<float> da_ref = {0.2866, 0.3764, 0.4038, 0.5779, 0.6875, 0.9335};
    std::vector<float> db_ref = {0.3738, 0.3854, 0.0977, 0.1171, 0.4386, 0.4390};

    SECTION("Dense") {
        Matmul matmul(&a, &b, &c, 2, 3, 2);
        matmul.forward(true);
        REQUIRE(allclose(c.data, c_ref));
        c.grad = c.data;
        matmul.backward();
        REQUIRE(allclose(a.grad, da_ref));
        REQUIRE(allclose(b.grad, db_ref));
    }

    SECTION("Sparse") {
        SparseIndex sp;
        sp.indices = {0, 1, 2, 0, 1, 2};
        sp.indptr = {0, 3, 6};
        SparseMatmul sparse_matmul(&a, &b, &c, &sp, 2, 3, 2);
        sparse_matmul.forward(true);
        REQUIRE(allclose(c.data, c_ref));
        c.grad = c.data;
        sparse_matmul.backward();
        //REQUIRE(allclose(a.grad, da_ref));
        REQUIRE(allclose(b.grad, db_ref));
    }
}