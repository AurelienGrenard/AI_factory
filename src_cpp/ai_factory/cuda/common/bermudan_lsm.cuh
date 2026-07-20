#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::cuda::bermudan_lsm {

inline constexpr std::size_t kBasisSize = 4U;
inline constexpr std::size_t kStatCount = 15U;
inline constexpr std::size_t kMaxExercises = 8U;

__device__ __forceinline__ void laguerre_basis(double x, double* basis) {
    const double x2 = x * x;
    basis[0] = 1.0;
    basis[1] = 1.0 - x;
    basis[2] = 1.0 - 2.0 * x + 0.5 * x2;
    basis[3] = 1.0 - 3.0 * x + 1.5 * x2 - x2 * x / 6.0;
}

__device__ __forceinline__ bool solve_linear_4x4(
    double matrix[kBasisSize][kBasisSize],
    double rhs[kBasisSize],
    double* solution
) {
    for (std::size_t pivot = 0; pivot < kBasisSize; ++pivot) {
        std::size_t best = pivot;
        double best_abs = fabs(matrix[pivot][pivot]);
        for (std::size_t row = pivot + 1U; row < kBasisSize; ++row) {
            const double candidate = fabs(matrix[row][pivot]);
            if (candidate > best_abs) {
                best = row;
                best_abs = candidate;
            }
        }
        if (best_abs < 1.0e-14) {
            return false;
        }
        if (best != pivot) {
            for (std::size_t column = 0; column < kBasisSize; ++column) {
                const double swap = matrix[pivot][column];
                matrix[pivot][column] = matrix[best][column];
                matrix[best][column] = swap;
            }
            const double swap = rhs[pivot];
            rhs[pivot] = rhs[best];
            rhs[best] = swap;
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
    for (std::size_t index = 0; index < kBasisSize; ++index) {
        solution[index] = rhs[index];
    }
    return true;
}

static __global__ void initialize_cashflows_kernel(
    const int* exercise_counts,
    const double* immediate,
    const double* discounts,
    std::size_t row_count,
    std::size_t num_paths,
    double* cashflows
) {
    const std::size_t row = blockIdx.x;
    if (row >= row_count) return;
    for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
        const std::size_t exercise = static_cast<std::size_t>(exercise_counts[row] - 1);
        const std::size_t state_index = (row * kMaxExercises + exercise) * num_paths + path;
        cashflows[row * num_paths + path] = discounts[state_index] * immediate[state_index];
    }
}

static __global__ void regression_stats_kernel(
    const int* exercise_counts,
    const double* immediate,
    const double* basis_states,
    const double* discounts,
    const double* cashflows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t exercise,
    double* stats
) {
    const std::size_t row = blockIdx.x;
    if (row >= row_count) return;
    extern __shared__ double shared[];
    double local[kStatCount]{};
    if (exercise + 1U < static_cast<std::size_t>(exercise_counts[row])) {
        for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
            const std::size_t state_index = (row * kMaxExercises + exercise) * num_paths + path;
            const double exercise_value = immediate[state_index];
            if (exercise_value <= 0.0) continue;
            double basis[kBasisSize];
            laguerre_basis(basis_states[state_index], basis);
            const double target = cashflows[row * num_paths + path] / discounts[state_index];
            std::size_t cursor = 0U;
            for (std::size_t i = 0; i < kBasisSize; ++i) {
                for (std::size_t j = i; j < kBasisSize; ++j) {
                    local[cursor++] += basis[i] * basis[j];
                }
            }
            for (std::size_t i = 0; i < kBasisSize; ++i) local[cursor++] += basis[i] * target;
            local[cursor] += 1.0;
        }
    }
    for (std::size_t stat = 0; stat < kStatCount; ++stat) {
        shared[stat * blockDim.x + threadIdx.x] = local[stat];
    }
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            for (std::size_t stat = 0; stat < kStatCount; ++stat) {
                shared[stat * blockDim.x + threadIdx.x] +=
                    shared[stat * blockDim.x + threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (std::size_t stat = 0; stat < kStatCount; ++stat) {
            stats[row * kStatCount + stat] = shared[stat * blockDim.x];
        }
    }
}

static __global__ void solve_kernel(
    const double* stats, std::size_t row_count, double* coefficients
) {
    const std::size_t row = blockIdx.x;
    if (row >= row_count || threadIdx.x != 0) return;
    const double* values = stats + row * kStatCount;
    double matrix[kBasisSize][kBasisSize]{};
    double rhs[kBasisSize]{};
    std::size_t cursor = 0U;
    for (std::size_t i = 0; i < kBasisSize; ++i) {
        for (std::size_t j = i; j < kBasisSize; ++j) {
            matrix[i][j] = matrix[j][i] = values[cursor++];
        }
    }
    for (std::size_t i = 0; i < kBasisSize; ++i) rhs[i] = values[cursor++];
    double solution[kBasisSize]{};
    const bool valid = values[cursor] > static_cast<double>(kBasisSize)
        && solve_linear_4x4(matrix, rhs, solution);
    for (std::size_t i = 0; i < kBasisSize; ++i) {
        coefficients[row * (kBasisSize + 1U) + i] = valid ? solution[i] : 0.0;
    }
    coefficients[row * (kBasisSize + 1U) + kBasisSize] = valid ? 1.0 : 0.0;
}

static __global__ void apply_exercise_kernel(
    const int* exercise_counts,
    const double* immediate,
    const double* basis_states,
    const double* discounts,
    const double* coefficients,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t exercise,
    double* cashflows
) {
    const std::size_t row = blockIdx.x;
    if (row >= row_count || exercise + 1U >= static_cast<std::size_t>(exercise_counts[row])) return;
    const double* coefficient = coefficients + row * (kBasisSize + 1U);
    if (coefficient[kBasisSize] <= 0.0) return;
    for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
        const std::size_t state_index = (row * kMaxExercises + exercise) * num_paths + path;
        const double exercise_value = immediate[state_index];
        if (exercise_value <= 0.0) continue;
        double basis[kBasisSize];
        laguerre_basis(basis_states[state_index], basis);
        double continuation = 0.0;
        for (std::size_t i = 0; i < kBasisSize; ++i) continuation += coefficient[i] * basis[i];
        if (exercise_value > continuation) {
            cashflows[row * num_paths + path] = discounts[state_index] * exercise_value;
        }
    }
}

static __global__ void finalize_kernel(
    const double* cashflows,
    std::size_t row_count,
    std::size_t num_paths,
    MonteCarloOutput* outputs
) {
    const std::size_t row = blockIdx.x;
    if (row >= row_count) return;
    extern __shared__ double shared[];
    double sum = 0.0, sumsq = 0.0;
    for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
        const double value = cashflows[row * num_paths + path];
        sum += value;
        sumsq += value * value;
    }
    shared[threadIdx.x] = sum;
    shared[blockDim.x + threadIdx.x] = sumsq;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
            shared[blockDim.x + threadIdx.x] += shared[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double n = static_cast<double>(num_paths);
        const double mean = shared[0] / n;
        const double variance = (shared[blockDim.x] - n * mean * mean) / (n - 1.0);
        outputs[row] = {mean, sqrt(fmax(variance, 0.0) / n)};
    }
}

}  // namespace ai_factory::cuda::bermudan_lsm
