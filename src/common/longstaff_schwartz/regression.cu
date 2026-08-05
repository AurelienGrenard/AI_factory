// Fixed-basis normal-equation accumulation and solve implementation.
#include "common/longstaff_schwartz/regression.cuh"

#include "common/longstaff_schwartz/linear_solver.cu"
#include "common/reductions.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz {

__device__ __forceinline__ void accumulate_normal_equations(
    const RegressionBasis& basis,
    double target,
    double (&values)[RegressionBasis::kRegressionValueCount]
) {
    std::size_t value_index = 0U;
    #pragma unroll
    for (std::size_t basis_row = 0U;
         basis_row < RegressionBasis::kSize;
         ++basis_row) {
        const double row_value = static_cast<double>(basis.values[basis_row]);
        #pragma unroll
        for (std::size_t basis_column = basis_row;
             basis_column < RegressionBasis::kSize;
             ++basis_column) {
            values[value_index++] +=
                row_value * static_cast<double>(basis.values[basis_column]);
        }
        values[RegressionBasis::kGramValueCount + basis_row] +=
            row_value * target;
    }
    values[RegressionBasis::kRegressionValueCount - 1U] += 1.0;
}

__device__ __forceinline__ void reduce_and_store_regression_partials(
    const double (&values)[RegressionBasis::kRegressionValueCount],
    std::size_t batch_price,
    std::size_t block_index,
    std::size_t blocks_per_price,
    double* regression_partials
) {
    const double* const totals = reductions::reduce_block_values(values);
    if (threadIdx.x != 0U) return;

    #pragma unroll
    for (std::size_t statistic = 0U;
         statistic < RegressionBasis::kRegressionValueCount;
         ++statistic) {
        regression_partials[
            (
                batch_price * RegressionBasis::kRegressionValueCount
                + statistic
            ) * blocks_per_price + block_index
        ] = totals[statistic];
    }
}

__device__ __forceinline__ void solve_regression_for_row(
    std::uint32_t exercise_count,
    std::uint32_t backward_level,
    std::size_t batch_price,
    std::size_t blocks_per_price,
    const double* regression_partials,
    double* regression_coefficients,
    std::uint32_t* regression_valid
) {
    if (backward_level >= exercise_count) return;

    double values[RegressionBasis::kRegressionValueCount] = {};
    #pragma unroll
    for (std::size_t statistic = 0U;
         statistic < RegressionBasis::kRegressionValueCount;
         ++statistic) {
        const double* const partials =
            regression_partials
            + (
                batch_price * RegressionBasis::kRegressionValueCount
                + statistic
            ) * blocks_per_price;
        for (std::size_t partial = threadIdx.x;
             partial < blocks_per_price;
             partial += blockDim.x) {
            values[statistic] += partials[partial];
        }
    }

    const double* const totals = reductions::reduce_block_values(values);
    if (threadIdx.x != 0U) return;

    double gram[RegressionBasis::kSize * RegressionBasis::kSize] = {};
    double rhs[RegressionBasis::kSize] = {};
    double* const coefficients =
        regression_coefficients + batch_price * RegressionBasis::kSize;
    std::size_t value_index = 0U;
    double trace = 0.0;

    for (std::size_t row_index = 0U;
         row_index < RegressionBasis::kSize;
         ++row_index) {
        for (std::size_t column = row_index;
             column < RegressionBasis::kSize;
             ++column) {
            const double value = totals[value_index++];
            gram[row_index * RegressionBasis::kSize + column] = value;
            gram[column * RegressionBasis::kSize + row_index] = value;
            if (row_index == column) trace += value;
        }
        rhs[row_index] = totals[
            RegressionBasis::kGramValueCount + row_index
        ];
    }

    const double ridge =
        kRidgeRelative * trace / static_cast<double>(RegressionBasis::kSize);
    for (std::size_t index = 0U;
         index < RegressionBasis::kSize;
         ++index) {
        gram[index * RegressionBasis::kSize + index] += ridge;
    }
    const double itm_count =
        totals[RegressionBasis::kRegressionValueCount - 1U];
    const bool solved =
        itm_count > static_cast<double>(RegressionBasis::kSize)
        && cholesky_solve_normal_equations(
            gram,
            rhs,
            coefficients,
            RegressionBasis::kSize,
            kCholeskyDiagonalFloor
        );
    regression_valid[batch_price] = solved ? 1U : 0U;
    if (!solved) {
        for (std::size_t index = 0U;
             index < RegressionBasis::kSize;
             ++index) {
            coefficients[index] = 0.0;
        }
    }
}

}  // namespace ai_factory::workbench::longstaff_schwartz
