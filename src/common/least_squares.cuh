// Small CUDA least-squares solvers shared by early-exercise products.
#pragma once

#include <cstddef>

namespace ai_factory::workbench::least_squares {

// Six-term Heston basis for normalized spot and instantaneous variance.
struct TwoFactorLaguerreBasis {
    static constexpr std::size_t kSize = 6U;
    static constexpr std::size_t kGramValueCount =
        kSize * (kSize + 1U) / 2U;
    static constexpr std::size_t kRegressionValueCount =
        kGramValueCount + kSize + 1U;

    float values[kSize];

    // Evaluate {1, L1(x), L2(x), z, z^2, L1(x)z}.
    __device__ __forceinline__ static TwoFactorLaguerreBasis evaluate(
        float primary_state,
        float secondary_state
    );
};

// Evaluate the Laguerre polynomial of degree zero.
__device__ __forceinline__ float laguerre_0(float value);

// Evaluate the Laguerre polynomial of degree one.
__device__ __forceinline__ float laguerre_1(float value);

// Evaluate the Laguerre polynomial of degree two.
__device__ __forceinline__ float laguerre_2(float value);

// Solve regularized normal equations in place with a Cholesky factorization.
__device__ __forceinline__ bool cholesky_solve_normal_equations(
    double* gram,
    const double* right_hand_side,
    double* coefficients,
    std::size_t basis_size,
    double diagonal_floor
);

}  // namespace ai_factory::workbench::least_squares
