// Deterministic positive definite four-factor correlation examples.
#include "correlation/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    const std::array<const char*, 4> factors{
        "asset_0.spot", "asset_1.spot", "asset_1.variance", "rate.factor"};
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t i = 0U; i < 8U; ++i) {
        const double a = -0.35 + 0.10 * static_cast<double>(i);
        const double b = 0.16 - 0.04 * static_cast<double>(i);
        const double c = -0.55 + 0.10 * static_cast<double>(i);
        const double d = -0.18 + 0.05 * static_cast<double>(i);
        const double e = 0.14 - 0.04 * static_cast<double>(i);
        const double f = 0.12;
        const std::array<std::array<double, 4>, 4> lower{{
            {{1.0, 0.0, 0.0, 0.0}},
            {{a, std::sqrt(1.0-a*a), 0.0, 0.0}},
            {{b, c, std::sqrt(1.0-b*b-c*c), 0.0}},
            {{d, e, f, std::sqrt(1.0-d*d-e*e-f*f)}}
        }};
        std::vector<double> dense(16U, 0.0);
        nlohmann::ordered_json matrix = nlohmann::ordered_json::array();
        for (std::size_t row = 0U; row < 4U; ++row) {
            nlohmann::ordered_json values = nlohmann::ordered_json::array();
            for (std::size_t column = 0U; column < 4U; ++column) {
                double value = 0.0;
                for (std::size_t factor = 0U; factor < 4U; ++factor)
                    value += lower[row][factor] * lower[column][factor];
                dense[row*4U+column] = value;
                values.push_back(value);
            }
            matrix.push_back(std::move(values));
        }
        correlation::packed_cholesky(dense, 4U);
        rows.push_back({
            {"id", datasets::format_row_id(i)},
            {"factors", factors},
            {"matrix", std::move(matrix)}
        });
    }
    constexpr auto path =
        "datasets/correlation/bs_heston_vasicek_matrices_01.json";
    datasets::write_json_file(path, {
        {"database_id", "bs_heston_vasicek_matrices_01"},
        {"catalog", "catalog/correlation/bs_heston_vasicek_matrices_01"},
        {"url", "https://datasets.ai-factory.example/v2/correlation/bs_heston_vasicek_matrices_01.json"},
        {"row_count", rows.size()},
        {"correlations", rows}
    });
    correlation::load_matrices(path);
    datasets::write_generation_receipt(
        "catalog/correlation/bs_heston_vasicek_matrices_01/generation.yaml",
        rows.size(), {{"method", "normalized triangular factors"}}, 0.0, 0.0);
}
