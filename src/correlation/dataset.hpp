// Host-side correlation rows and their packed Cholesky factors.
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::correlation {

struct MatrixRow {
    std::string id;
    std::vector<std::string> factors;
    std::vector<float> lower; // row-major lower triangle, i*(i+1)/2+j
    std::size_t dimension() const noexcept { return factors.size(); }
};

// The input JSON has a "correlations" array. Each row has id, factors and
// a square row-major "matrix" array. The loader validates and factorizes R.
std::vector<MatrixRow> load_matrices(const std::filesystem::path& path);

// Reorder a loaded row by factor name and recompute its Cholesky.
// The original covariance is reconstructed from L L^T before permutation.
MatrixRow reorder_factors(
    const MatrixRow& row, const std::vector<std::string>& factor_order
);

// Validate a dense correlation matrix and return its packed FP32 Cholesky.
std::vector<float> packed_cholesky(
    const std::vector<double>& matrix, std::size_t dimension
);

}  // namespace ai_factory::workbench::correlation
