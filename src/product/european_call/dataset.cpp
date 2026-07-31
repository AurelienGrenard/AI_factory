// Convert European-call JSON rows into compact CUDA parameters.
#include "product/european_call/dataset.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<EuropeanCallInput> load_european_calls(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open European call JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid European call JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    const auto& rows = document.at("products");
    std::vector<EuropeanCallInput> products;
    products.reserve(rows.size());
    // Only strike and maturity are retained; JSON metadata stays out of CUDA.
    for (const auto& row : rows) {
        const auto& parameters = row.at("parameters");
        const EuropeanCallInput product = {
            parameters.at("strike").get<float>(),
            parameters.at("maturity").get<float>(),
        };
        if (!std::isfinite(product.strike)
            || !std::isfinite(product.maturity)
            || !(product.strike > 0.0f) || !(product.maturity > 0.0f)) {
            throw std::invalid_argument("Invalid European call input.");
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
