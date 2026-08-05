// Small dense linear solver used by Longstaff-Schwartz regressions.
#pragma once

#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz {

__device__ __forceinline__ bool cholesky_solve_normal_equations(
    double* gram,
    const double* right_hand_side,
    double* coefficients,
    std::size_t basis_size,
    double diagonal_floor
);

}  // namespace ai_factory::workbench::longstaff_schwartz
