// Small dense matrix exponential shared by host and device rough-Heston lift preparation.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace ai_factory::workbench::math {

template<std::size_t Dimension>
struct SmallMatrix {
    double values[Dimension * Dimension]{};

    __host__ __device__ double& operator[](std::size_t index) {
        return values[index];
    }
    __host__ __device__ const double& operator[](std::size_t index) const {
        return values[index];
    }
    __host__ __device__ double* begin() { return values; }
    __host__ __device__ double* end() { return values + Dimension * Dimension; }
    __host__ __device__ constexpr std::size_t size() const {
        return Dimension * Dimension;
    }
};

template<std::size_t Dimension>
__host__ __device__ SmallMatrix<Dimension> identity_matrix() {
    SmallMatrix<Dimension> result{};
    for (std::size_t index = 0U; index < Dimension; ++index)
        result[index * Dimension + index] = 1.0;
    return result;
}

template<std::size_t Dimension>
__host__ __device__ double matrix_one_norm(
    const SmallMatrix<Dimension>& matrix
) {
    double norm = 0.0;
    for (std::size_t column = 0U; column < Dimension; ++column) {
        double sum = 0.0;
        for (std::size_t row = 0U; row < Dimension; ++row)
            sum += fabs(matrix[row * Dimension + column]);
        norm = fmax(norm, sum);
    }
    return norm;
}

template<std::size_t Dimension>
__host__ __device__ SmallMatrix<Dimension> multiply(
    const SmallMatrix<Dimension>& left,
    const SmallMatrix<Dimension>& right
) {
    SmallMatrix<Dimension> result{};
    for (std::size_t row = 0U; row < Dimension; ++row) {
        for (std::size_t column = 0U; column < Dimension; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0U; inner < Dimension; ++inner) {
                value += left[row * Dimension + inner]
                    * right[inner * Dimension + column];
            }
            result[row * Dimension + column] = value;
        }
    }
    return result;
}

// Scaling and squaring with a converged Taylor series for Dimension <= 8.
template<std::size_t Dimension>
__host__ __device__ SmallMatrix<Dimension> matrix_exponential(
    SmallMatrix<Dimension> matrix
) {
    const double norm = matrix_one_norm(matrix);
    int scaling = 0;
    if (norm > 0.5)
        scaling = static_cast<int>(ceil(log2(norm / 0.5)));
    const double divisor = ldexp(1.0, scaling);
    for (double& value : matrix) value /= divisor;

    SmallMatrix<Dimension> result = identity_matrix<Dimension>();
    SmallMatrix<Dimension> term = result;
    for (int order = 1; order <= 64; ++order) {
        term = multiply(term, matrix);
        const double inverse_order = 1.0 / static_cast<double>(order);
        for (double& value : term) value *= inverse_order;
        for (std::size_t index = 0U; index < result.size(); ++index)
            result[index] += term[index];
        if (matrix_one_norm(term) < 2.0e-16) break;
    }
    for (int square = 0; square < scaling; ++square)
        result = multiply(result, result);
    return result;
}

}  // namespace ai_factory::workbench::math
