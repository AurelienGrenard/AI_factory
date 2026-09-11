// Compile-time small-basis normal-equation regression for LSM kernels.
#pragma once

#include "common/longstaff_schwartz/linear_solver.cuh"
#include "common/longstaff_schwartz/regression_status.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>

namespace ai_factory::workbench::longstaff_schwartz {

inline constexpr std::size_t kMaximumSmallRegressionBasisSize = 8U;
inline constexpr double kRidgeRelative = 1.0e-10;
inline constexpr double kCholeskyDiagonalFloor = 1.0e-14;

enum class RegressionRefinement { none, normal_residual };

template<
    typename BasisPolicy,
    RegressionRefinement Refinement = RegressionRefinement::none
>
struct NormalEquationRegressor {
    using Basis = BasisPolicy;
    using Input = typename Basis::Input;
    using Features = typename Basis::Features;

    static constexpr std::size_t kBasisSize = Basis::kSize;
    static constexpr std::size_t kGramValueCount =
        kBasisSize * (kBasisSize + 1U) / 2U;
    static constexpr std::size_t kRegressionValueCount =
        kGramValueCount + kBasisSize + 1U;
    static constexpr bool kRefineNormalResidual =
        Refinement == RegressionRefinement::normal_residual;

    static_assert(kBasisSize > 0U);
    static_assert(
        kBasisSize <= kMaximumSmallRegressionBasisSize,
        "The small Longstaff-Schwartz regressor supports at most eight "
        "basis functions; use a different kernel strategy for larger bases."
    );
    static_assert(Features::kSize == kBasisSize);

    __device__ __forceinline__ static Features evaluate(const Input& input) {
        return Basis::evaluate(input);
    }

    __device__ __forceinline__ static void accumulate(
        const Features& features,
        double target,
        double (&values)[kRegressionValueCount]
    ) {
        std::size_t value_index = 0U;
        #pragma unroll
        for (std::size_t basis_row = 0U;
             basis_row < kBasisSize;
             ++basis_row) {
            const double row_value =
                static_cast<double>(features.values[basis_row]);
            #pragma unroll
            for (std::size_t basis_column = basis_row;
                 basis_column < kBasisSize;
                 ++basis_column) {
                values[value_index++] += row_value * static_cast<double>(
                    features.values[basis_column]
                );
            }
            values[kGramValueCount + basis_row] += row_value * target;
        }
        values[kRegressionValueCount - 1U] += 1.0;
    }

    __device__ __forceinline__ static void reduce_and_store_partials(
        const double (&values)[kRegressionValueCount],
        std::size_t batch_price,
        std::size_t block_index,
        std::size_t blocks_per_price,
        double* regression_partials
    ) {
        const double* const totals = reductions::reduce_block_values(values);
        if (threadIdx.x != 0U) return;

        #pragma unroll
        for (std::size_t statistic = 0U;
             statistic < kRegressionValueCount;
             ++statistic) {
            regression_partials[
                (batch_price * kRegressionValueCount
                    + statistic) * blocks_per_price
                + block_index
            ] = totals[statistic];
        }
    }

    // One residual pass retains the original basis, ridge and Gram. Computing
    // Phi^T(y - Phi*beta) from observations avoids subtracting rounded G*beta.
    __device__ __forceinline__ static void accumulate_residual(
        const Features& features,
        double target,
        const double* coefficients,
        double (&values)[kBasisSize]
    ) {
        double residual = target;
        #pragma unroll
        for (std::size_t index = 0U; index < kBasisSize; ++index) {
            residual = fma(
                -coefficients[index],
                static_cast<double>(features.values[index]),
                residual
            );
        }
        #pragma unroll
        for (std::size_t index = 0U; index < kBasisSize; ++index) {
            values[index] += static_cast<double>(features.values[index])
                * residual;
        }
    }

    __device__ __forceinline__ static void reduce_and_store_residual_partials(
        const double (&values)[kBasisSize],
        std::size_t batch_price,
        std::size_t block_index,
        std::size_t blocks_per_price,
        double* regression_partials
    ) {
        extern __shared__ double shared[];
        const unsigned int lane = threadIdx.x & 31U;
        const unsigned int warp = threadIdx.x >> 5U;
        const unsigned int warp_count = blockDim.x >> 5U;
        #pragma unroll
        for (std::size_t index = 0U; index < kBasisSize; ++index) {
            double sum = values[index];
            double error = 0.0;
            reduce_residual_warp(sum, error);
            if (lane == 0U) {
                shared[index * warp_count + warp] = sum;
                shared[(index + kBasisSize) * warp_count + warp] = error;
            }
        }
        __syncthreads();
        if (warp == 0U) {
            #pragma unroll
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                double sum = lane < warp_count
                    ? shared[index * warp_count + lane] : 0.0;
                double error = lane < warp_count
                    ? shared[(index + kBasisSize) * warp_count + lane] : 0.0;
                reduce_residual_warp(sum, error);
                if (lane == 0U) {
                    // Reuse RHS slots only; Gram and sample count survive.
                    regression_partials[(batch_price * kRegressionValueCount
                        + kGramValueCount + index) * blocks_per_price
                        + block_index] = sum + error;
                }
            }
        }
    }

    template<bool ApplyCorrection = false>
    __device__ __forceinline__ static void solve_for_row(
        std::uint32_t regression_count,
        std::uint32_t backward_level,
        std::size_t batch_price,
        std::size_t blocks_per_price,
        const double* regression_partials,
        double* regression_coefficients,
        RegressionStatus* regression_status,
        RegressionDiagnostics* regression_diagnostics
    ) {
        if (backward_level >= regression_count) return;
        if constexpr (ApplyCorrection) {
            if (regression_status[batch_price] != RegressionStatus::success) {
                return;
            }
        }

        double values[kRegressionValueCount] = {};
        #pragma unroll
        for (std::size_t statistic = 0U;
             statistic < kRegressionValueCount;
             ++statistic) {
            const double* const partials = regression_partials + (
                batch_price * kRegressionValueCount + statistic
            ) * blocks_per_price;
            for (std::size_t partial = threadIdx.x;
                 partial < blocks_per_price;
                 partial += blockDim.x) {
                values[statistic] += partials[partial];
            }
        }

        const double* const totals = reductions::reduce_block_values(values);
        if (threadIdx.x != 0U) return;

        double* const coefficients =
            regression_coefficients + batch_price * kBasisSize;
        double previous_coefficients[kBasisSize] = {};
        if constexpr (ApplyCorrection) {
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                previous_coefficients[index] = coefficients[index];
            }
        }
        RegressionStatus status = RegressionStatus::success;
        const double sample_count = totals[kRegressionValueCount - 1U];
        #pragma unroll
        for (std::size_t statistic = 0U;
             statistic < kRegressionValueCount;
             ++statistic) {
            if (!isfinite(totals[statistic])) {
                status = RegressionStatus::non_finite_statistics;
            }
        }
        if (status == RegressionStatus::success && sample_count == 0.0) {
            status = RegressionStatus::no_candidates;
        } else if (status == RegressionStatus::success
            && sample_count <= static_cast<double>(kBasisSize)) {
            status = RegressionStatus::insufficient_candidates;
        }

        double gram[kBasisSize * kBasisSize] = {};
        double right_hand_side[kBasisSize] = {};
        std::size_t value_index = 0U;
        double trace = 0.0;

        for (std::size_t row = 0U; row < kBasisSize; ++row) {
            for (std::size_t column = row;
                 column < kBasisSize;
                 ++column) {
                const double value = totals[value_index++];
                gram[row * kBasisSize + column] = value;
                gram[column * kBasisSize + row] = value;
                if (row == column) trace += value;
            }
            right_hand_side[row] = totals[kGramValueCount + row];
        }

        if (status == RegressionStatus::success) {
            const double ridge = kRidgeRelative * trace
                / static_cast<double>(kBasisSize);
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                gram[index * kBasisSize + index] += ridge;
                if constexpr (ApplyCorrection) {
                    right_hand_side[index] -= ridge
                        * previous_coefficients[index];
                }
            }
            if (!cholesky_solve_normal_equations(
                gram,
                right_hand_side,
                coefficients,
                kBasisSize,
                kCholeskyDiagonalFloor
            )) {
                status = RegressionStatus::factorization_failure;
            }
        }
        if constexpr (ApplyCorrection) {
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                coefficients[index] += previous_coefficients[index];
            }
        }
        if (status == RegressionStatus::success) {
            #pragma unroll
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                if (!isfinite(coefficients[index])) {
                    status = RegressionStatus::non_finite_coefficients;
                }
            }
        }
        regression_status[batch_price] = status;
        // Exactly one diagnostic per date, including a failed correction.
        if constexpr (!kRefineNormalResidual || ApplyCorrection) {
            record_regression_status(status, regression_diagnostics[batch_price]);
        } else if (status != RegressionStatus::success) {
            record_regression_status(status, regression_diagnostics[batch_price]);
        }
        if (status != RegressionStatus::success) {
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                coefficients[index] = 0.0;
            }
        }
    }

    __device__ __forceinline__ static double predict(
        const Features& features,
        const double* coefficients
    ) {
        double continuation = 0.0;
        #pragma unroll
        for (std::size_t index = 0U; index < kBasisSize; ++index) {
            continuation = fma(
                coefficients[index],
                static_cast<double>(features.values[index]),
                continuation
            );
        }
        return continuation;
    }

    inline static std::size_t shared_bytes(unsigned int threads_per_block) {
        const std::size_t warp_count = threads_per_block / 32U;
        return (
            kRegressionValueCount * warp_count + kRegressionValueCount
        ) * sizeof(double);
    }

private:
    // Error-free TwoSum at reduction edges only; path-local sums remain FP64.
    __device__ __forceinline__ static void reduce_residual_warp(
        double& sum,
        double& error
    ) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double next = __shfl_down_sync(0xFFFFFFFFU, sum, offset);
            const double next_error = __shfl_down_sync(0xFFFFFFFFU, error, offset);
            const double previous = sum;
            sum = __dadd_rn(previous, next);
            const double virtual_next = __dsub_rn(sum, previous);
            error += __dadd_rn(
                __dsub_rn(previous, __dsub_rn(sum, virtual_next)),
                __dsub_rn(next, virtual_next)
            );
            error += next_error;
        }
    }
};

}  // namespace ai_factory::workbench::longstaff_schwartz
