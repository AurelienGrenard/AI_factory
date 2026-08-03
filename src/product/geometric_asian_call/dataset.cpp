// Convert Geometric-Asian-call JSON rows into compact CUDA parameters.
#include "product/geometric_asian_call/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<GeometricAsianCallParameters> load_geometric_asian_calls(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Geometric Asian call JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Geometric Asian call JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<GeometricAsianCallParameters> products;
    products.reserve(rows.size());
    // Only strike and maturity are retained; JSON metadata stays out of CUDA.
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const GeometricAsianCallParameters product = {
            parameters.at("strike").get<float>(),
            parameters.at("maturity").get<float>(),
        };
        const std::string prefix = "Geometric Asian call row id '" + row_id + "': ";
        if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
            throw std::invalid_argument(prefix + "strike must be finite and positive.");
        if (!std::isfinite(product.maturity) || !(product.maturity > 0.0f))
            throw std::invalid_argument(prefix + "maturity must be finite and positive.");
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
