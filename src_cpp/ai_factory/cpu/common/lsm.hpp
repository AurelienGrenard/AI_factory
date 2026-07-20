#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace ai_factory::cpu::lsm {

inline constexpr std::size_t kBasisSize = 4U;
using Basis = std::array<double, kBasisSize>;
using Matrix = std::array<Basis, kBasisSize>;

inline Basis laguerre_basis(double x) {
    const double x2 = x * x;
    return {
        1.0,
        1.0 - x,
        1.0 - 2.0 * x + 0.5 * x2,
        1.0 - 3.0 * x + 1.5 * x2 - x2 * x / 6.0,
    };
}

inline bool solve_linear_4x4(
    Matrix matrix,
    Basis rhs,
    Basis& solution
) {
    for (std::size_t pivot = 0; pivot < kBasisSize; ++pivot) {
        std::size_t best = pivot;
        double best_abs = std::abs(matrix[pivot][pivot]);
        for (std::size_t row = pivot + 1U; row < kBasisSize; ++row) {
            const double candidate = std::abs(matrix[row][pivot]);
            if (candidate > best_abs) {
                best = row;
                best_abs = candidate;
            }
        }
        if (best_abs < 1.0e-14) {
            return false;
        }
        if (best != pivot) {
            std::swap(matrix[pivot], matrix[best]);
            std::swap(rhs[pivot], rhs[best]);
        }
        const double diagonal = matrix[pivot][pivot];
        for (std::size_t column = pivot; column < kBasisSize; ++column) {
            matrix[pivot][column] /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for (std::size_t row = 0; row < kBasisSize; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = matrix[row][pivot];
            for (std::size_t column = pivot; column < kBasisSize; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    solution = rhs;
    return true;
}

}  // namespace ai_factory::cpu::lsm
