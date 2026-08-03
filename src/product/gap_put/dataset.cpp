// Convert gap-put JSON rows into compact CUDA parameters.
#include "product/gap_put/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<GapPutParameters> load_gap_puts(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open gap put JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid gap put JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<GapPutParameters> products;
    products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const std::string prefix = "Gap put row id '" + row_id + "': ";
        const GapPutParameters product = {
            parameters.at("trigger_strike").get<float>(),
            parameters.at("payoff_strike").get<float>(),
            parameters.at("maturity").get<float>(),
        };
        if (!std::isfinite(product.trigger_strike)
            || !(product.trigger_strike > 0.0f)) {
            throw std::invalid_argument(
                prefix + "trigger_strike must be finite and positive."
            );
        }
        if (!std::isfinite(product.payoff_strike)
            || !(product.payoff_strike > 0.0f)) {
            throw std::invalid_argument(
                prefix + "payoff_strike must be finite and positive."
            );
        }
        if (!std::isfinite(product.maturity) || !(product.maturity > 0.0f))
            throw std::invalid_argument(prefix + "maturity must be finite and positive.");
        if (product.payoff_strike < product.trigger_strike) {
            throw std::invalid_argument(
                prefix + "payoff_strike must not be below trigger_strike."
            );
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
