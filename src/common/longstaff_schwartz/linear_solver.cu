// Cholesky factorization for the small dense LSM normal equations.
#include "common/longstaff_schwartz/linear_solver.cuh"

#include <cmath>

namespace ai_factory::workbench::longstaff_schwartz {

__device__ __forceinline__ bool cholesky_solve_normal_equations(
    double* gram,
    const double* right_hand_side,
    double* coefficients,
    std::size_t basis_size,
    double diagonal_floor
) {
    if (basis_size == 0U || !(diagonal_floor > 0.0)) return false;

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

    for (std::size_t row = 0U; row < basis_size; ++row) {
        double value = right_hand_side[row];
        for (std::size_t column = 0U; column < row; ++column) {
            value -= gram[row * basis_size + column] * coefficients[column];
        }
        coefficients[row] = value / gram[row * basis_size + row];
    }

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

}  // namespace ai_factory::workbench::longstaff_schwartz
