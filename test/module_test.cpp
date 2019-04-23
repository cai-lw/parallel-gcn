#include <catch2/catch.hpp>
#include "../src/module.h"
#include "util.h"
#include <cmath>

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
        matmul.forward(false);
        REQUIRE(allclose(c.data, c_ref));
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
        sparse_matmul.forward(false);
        REQUIRE(allclose(c.data, c_ref));
    }
}

TEST_CASE("Cross entropy") {
    Variable a(9);
    a.data = {0.9060, 0.8809, 0.0622, 0.1152, 0.9809, 0.6692, 0.9106, 0.3262, 0.5933};
    std::vector<float> da_ref = {-0.1947, 0.1351, 0.0596, 0.0651, -0.1785, 0.1134, 0.1458, 0.0813, -0.2271};
    std::vector<int> truth{0, 1, 2};
    float loss;
    CrossEntropyLoss cel(&a, truth.data(), &loss, 3);
    cel.forward(true);
    REQUIRE(abs(loss - 0.9295) < 1e-3);
    cel.backward();
    REQUIRE(allclose(a.grad, da_ref));
    cel.forward(false);
    REQUIRE(abs(loss - 0.9295) < 1e-3);
}

TEST_CASE("Graph sum") {
    SparseIndex graph;                          // 0 - 1
    graph.indices = {1, 2, 3, 0, 2, 0, 1, 0};   // | \ |
    graph.indptr = {0, 3, 5, 7, 8};             // 3   2

    Variable a(8), b(8);
    std::vector<float> a_ref = {4.0, 3.0, 2.0, 1.0};
    std::vector<float> b_ref = {
        3.0f / sqrtf(6) + 2.0f / sqrtf(6) + 1.0f / sqrtf(3),
        4.0f / sqrtf(6) + 2.0f / sqrtf(4),
        4.0f / sqrtf(6) + 3.0f / sqrtf(4),
        4.0f / sqrtf(3)
    };
    a.data = a_ref;

    GraphSum graphsum(&a, &b, &graph, 1);
    graphsum.forward(true);
    REQUIRE(allclose(b.data, b_ref));
    b.grad = a_ref;
    graphsum.backward();
    REQUIRE(allclose(a.grad, b_ref));
    graphsum.forward(false);
    REQUIRE(allclose(b.data, b_ref));
}
