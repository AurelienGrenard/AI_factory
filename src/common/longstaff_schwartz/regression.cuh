// Fixed-basis normal-equation accumulation and solve helpers for LSM kernels.
#pragma once

#include "common/longstaff_schwartz/laguerre.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz {

using RegressionBasis = TwoFactorLaguerreBasis;

constexpr double kRidgeRelative = 1.0e-10;
constexpr double kCholeskyDiagonalFloor = 1.0e-14;

__device__ __forceinline__ void accumulate_normal_equations(
    const RegressionBasis& basis,
    double target,
    double (&values)[RegressionBasis::kRegressionValueCount]
);

__device__ __forceinline__ void reduce_and_store_regression_partials(
    const double (&values)[RegressionBasis::kRegressionValueCount],
    std::size_t batch_price,
    std::size_t block_index,
    std::size_t blocks_per_price,
    double* regression_partials
);

__device__ __forceinline__ void solve_regression_for_row(
    std::uint32_t exercise_count,
    std::uint32_t backward_level,
    std::size_t batch_price,
    std::size_t blocks_per_price,
    const double* regression_partials,
    double* regression_coefficients,
    std::uint32_t* regression_valid
);

inline std::size_t regression_shared_bytes(unsigned int threads_per_block) {
    const std::size_t warp_count = threads_per_block / 32U;
    return (
        RegressionBasis::kRegressionValueCount * warp_count
        + RegressionBasis::kRegressionValueCount
    ) * sizeof(double);
}

}  // namespace ai_factory::workbench::longstaff_schwartz
