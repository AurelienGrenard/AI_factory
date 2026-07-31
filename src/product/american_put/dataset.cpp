// Convert American-put JSON rows into compact CUDA parameters.
#include "product/american_put/dataset.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<AmericanPutParameters> load_american_puts(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open American put JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid American put JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    const auto& rows = document.at("products");
    std::vector<AmericanPutParameters> products;
    products.reserve(rows.size());
    // Retain only the three FP32 fields required by pricing.
    for (const auto& row : rows) {
        const auto& parameters = row.at("parameters");
        const AmericanPutParameters product = {
            parameters.at("strike").get<float>(),
            parameters.at("maturity").get<float>(),
            parameters.at("exercise_interval").get<float>(),
        };
        if (!std::isfinite(product.strike)
            || !std::isfinite(product.maturity)
            || !std::isfinite(product.exercise_interval)
            || !(product.strike > 0.0f)
            || !(product.maturity > 0.0f)
            || !(product.exercise_interval > 0.0f)
            || !(product.exercise_interval < product.maturity)) {
            throw std::invalid_argument("Invalid American put input.");
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
