// Convert Forward-start-put JSON rows into compact CUDA parameters.
#include "product/forward_start_put/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<ForwardStartPutParameters> load_forward_start_puts(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Forward-start put JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Forward-start put JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<ForwardStartPutParameters> products;
    products.reserve(rows.size());
    // Retain only the reset contract fields consumed by CUDA.
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const ForwardStartPutParameters product = {
            parameters.at("moneyness").get<float>(),
            parameters.at("reset_time").get<float>(),
            parameters.at("maturity").get<float>(),
        };
        const std::string prefix =
            "Forward-start put row id '" + row_id + "': ";
        if (!std::isfinite(product.moneyness) || !(product.moneyness > 0.0f))
            throw std::invalid_argument(prefix + "moneyness must be finite and positive.");
        if (!std::isfinite(product.reset_time) || !(product.reset_time > 0.0f))
            throw std::invalid_argument(prefix + "reset_time must be finite and positive.");
        if (!std::isfinite(product.maturity) || !(product.maturity > 0.0f))
            throw std::invalid_argument(prefix + "maturity must be finite and positive.");
        if (!(product.reset_time < product.maturity))
            throw std::invalid_argument(prefix + "reset_time must precede maturity.");
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
