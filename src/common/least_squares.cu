// Cholesky implementation for the small normal equations of LSM regressions.
#include "common/least_squares.cuh"

#include <cmath>

namespace ai_factory::workbench::least_squares {

// Degree zero anchors every Laguerre regression with a constant.
__device__ __forceinline__ float laguerre_0(float) {
    return 1.0f;
}

// First Laguerre polynomial.
__device__ __forceinline__ float laguerre_1(float value) {
    return 1.0f - value;
}

// Second Laguerre polynomial evaluated with fused operations.
__device__ __forceinline__ float laguerre_2(float value) {
    return fmaf(0.5f * value, value, 1.0f - 2.0f * value);
}

// Couple a quadratic spot state with a quadratic variance state.
__device__ __forceinline__ TwoFactorLaguerreBasis
TwoFactorLaguerreBasis::evaluate(
    float primary_state,
    float secondary_state
) {
    const float primary_1 = laguerre_1(primary_state);
    return {{
        laguerre_0(primary_state),
        primary_1,
        laguerre_2(primary_state),
        secondary_state,
        secondary_state * secondary_state,
        primary_1 * secondary_state,
    }};
}

// Factor G = LL^T, then solve both triangular systems without forming G^-1.
__device__ __forceinline__ bool cholesky_solve_normal_equations(
    double* gram,
    const double* right_hand_side,
    double* coefficients,
    std::size_t basis_size,
    double diagonal_floor
) {
    if (basis_size == 0U || !(diagonal_floor > 0.0)) return false;

    // Overwrite the lower triangle of G with its Cholesky factor L.
    for (std::size_t row = 0U; row < basis_size; ++row) {
        for (std::size_t column = 0U; column <= row; ++column) {
            double value = gram[row * basis_size + column];
            for (std::size_t inner = 0U; inner < column; ++inner) {
                value -= gram[row * basis_size + inner]
                    * gram[column * basis_size + inner];
            }
            if (row == column) {
                if (!(value > diagonal_floor)) return false;
                gram[row * basis_size + column] = sqrt(value);
            } else {
                gram[row * basis_size + column] =
                    value / gram[column * basis_size + column];
            }
        }
    }

    // Forward substitution solves Lz = b into the coefficient workspace.
    for (std::size_t row = 0U; row < basis_size; ++row) {
        double value = right_hand_side[row];
        for (std::size_t column = 0U; column < row; ++column) {
            value -= gram[row * basis_size + column] * coefficients[column];
        }
        coefficients[row] = value / gram[row * basis_size + row];
    }

    // Backward substitution replaces z with the final coefficients.
    for (std::size_t row = basis_size; row-- > 0U;) {
        double value = coefficients[row];
        for (std::size_t column = row + 1U;
             column < basis_size;
             ++column) {
            value -= gram[column * basis_size + row] * coefficients[column];
        }
        coefficients[row] = value / gram[row * basis_size + row];
    }
    return true;
}

}  // namespace ai_factory::workbench::least_squares
