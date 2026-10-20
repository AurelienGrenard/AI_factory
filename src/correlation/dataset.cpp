#include "correlation/dataset.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <unordered_set>

namespace ai_factory::workbench::correlation {

std::vector<float> packed_cholesky(
    const std::vector<double>& matrix, std::size_t dimension
) {
    if (dimension == 0U || matrix.size() != dimension * dimension) {
        throw std::invalid_argument("Correlation matrix has invalid dimensions.");
    }
    constexpr double tolerance = 1.0e-6;
    for (std::size_t i = 0; i < dimension; ++i) {
        const double diagonal = matrix[i * dimension + i];
        if (!std::isfinite(diagonal) || std::abs(diagonal - 1.0) > tolerance) {
            throw std::invalid_argument("Correlation matrix diagonal must be one.");
        }
        for (std::size_t j = 0; j < i; ++j) {
            const double a = matrix[i * dimension + j];
            const double b = matrix[j * dimension + i];
            if (!std::isfinite(a) || !std::isfinite(b)
                || std::abs(a) > 1.0 + tolerance
                || std::abs(a - b) > tolerance) {
                throw std::invalid_argument("Correlation matrix is not symmetric.");
            }
        }
    }

    std::vector<double> lower(dimension * (dimension + 1U) / 2U);
    const auto index = [](std::size_t i, std::size_t j) {
        return i * (i + 1U) / 2U + j;
    };
    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = matrix[i * dimension + j];
            for (std::size_t k = 0; k < j; ++k) {
                value -= lower[index(i, k)] * lower[index(j, k)];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::invalid_argument(
                        "Correlation matrix must be positive definite."
                    );
                }
                lower[index(i, j)] = std::sqrt(value);
            } else {
                lower[index(i, j)] = value / lower[index(j, j)];
            }
        }
    }
    std::vector<float> result(lower.size());
    for (std::size_t i = 0; i < lower.size(); ++i) {
        result[i] = static_cast<float>(lower[i]);
        if (!std::isfinite(result[i])) {
            throw std::invalid_argument("Cholesky factor is not finite in FP32.");
        }
    }
    return result;
}

MatrixRow reorder_factors(
    const MatrixRow& row, const std::vector<std::string>& factor_order
) {
    const auto dimension = row.dimension();
    if (factor_order.size() != dimension
        || row.lower.size() != dimension * (dimension + 1U) / 2U) {
        throw std::invalid_argument("Correlation factor permutation has invalid dimensions.");
    }
    std::unordered_map<std::string, std::size_t> positions;
    for (std::size_t i = 0U; i < dimension; ++i) {
        if (!positions.emplace(row.factors[i], i).second) {
            throw std::invalid_argument("Correlation factor names must be unique.");
        }
    }
    std::vector<std::size_t> permutation;
    permutation.reserve(dimension);
    std::unordered_set<std::string> seen;
    for (const auto& factor : factor_order) {
        if (!seen.insert(factor).second || !positions.contains(factor)) {
            throw std::invalid_argument("Correlation permutation has unknown or duplicate factors.");
        }
        permutation.push_back(positions.at(factor));
    }
    std::vector<double> matrix(dimension * dimension);
    const auto index = [](std::size_t i, std::size_t j) {
        return i * (i + 1U) / 2U + j;
    };
    for (std::size_t i = 0U; i < dimension; ++i) {
        for (std::size_t j = 0U; j < dimension; ++j) {
            const auto left = permutation[i];
            const auto right = permutation[j];
            double covariance = 0.0;
            for (std::size_t k = 0U; k <= std::min(left, right); ++k) {
                covariance += static_cast<double>(row.lower[index(left, k)])
                    * static_cast<double>(row.lower[index(right, k)]);
            }
            matrix[i * dimension + j] = covariance;
        }
    }
    // FP32 Cholesky reconstruction may move the diagonal by a few ulps.
    for (std::size_t i = 0U; i < dimension; ++i) {
        matrix[i * dimension + i] = 1.0;
        for (std::size_t j = 0U; j < i; ++j) {
            const double average = 0.5 * (
                matrix[i * dimension + j] + matrix[j * dimension + i]
            );
            matrix[i * dimension + j] = average;
            matrix[j * dimension + i] = average;
        }
    }
    return {row.id, factor_order, packed_cholesky(matrix, dimension)};
}

std::vector<MatrixRow> load_matrices(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open correlation dataset: " + path.string());
    nlohmann::json document;
    stream >> document;
    const auto& rows = document.at("correlations");
    if (!rows.is_array() || rows.empty()) {
        throw std::invalid_argument("Correlation dataset must contain rows.");
    }
    if (!document.contains("row_count")
        || !document.at("row_count").is_number_integer()
        || document.at("row_count").get<std::size_t>() != rows.size()) {
        throw std::invalid_argument(
            "Correlation dataset row_count disagrees with its rows."
        );
    }
    std::vector<MatrixRow> output;
    output.reserve(rows.size());
    std::unordered_set<std::string> ids;
    for (const auto& row : rows) {
        MatrixRow result;
        result.id = row.at("id").get<std::string>();
        result.factors = row.at("factors").get<std::vector<std::string>>();
        if (result.id.empty() || !ids.insert(result.id).second
            || result.factors.empty()) {
            throw std::invalid_argument("Correlation rows need unique nonempty IDs and factors.");
        }
        std::unordered_set<std::string> factor_names;
        for (const auto& name : result.factors) {
            if (name.empty() || !factor_names.insert(name).second) {
                throw std::invalid_argument("Correlation factor names must be unique.");
            }
        }
        const auto& dense = row.at("matrix");
        const auto d = result.dimension();
        if (!dense.is_array() || dense.size() != d) {
            throw std::invalid_argument("Correlation row has invalid matrix dimensions.");
        }
        std::vector<double> values;
        values.reserve(d * d);
        for (const auto& matrix_row : dense) {
            if (!matrix_row.is_array() || matrix_row.size() != d) {
                throw std::invalid_argument("Correlation row has invalid matrix dimensions.");
            }
            for (const auto& value : matrix_row) {
                values.push_back(value.get<double>());
            }
        }
        result.lower = packed_cholesky(values, d);
        output.push_back(std::move(result));
    }
    return output;
}

}  // namespace ai_factory::workbench::correlation
