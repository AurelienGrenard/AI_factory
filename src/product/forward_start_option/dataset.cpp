// Convert Forward-start-option JSON rows into compact CUDA parameters.
#include "product/forward_start_option/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<ForwardStartOptionParameters> load_forward_start_options(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Forward-start option JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Forward-start option JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<ForwardStartOptionParameters> products;
    products.reserve(rows.size());
    // Retain only the reset contract fields consumed by CUDA.
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const ForwardStartOptionParameters product = {
            parameters.at("moneyness").get<float>(),
            parameters.at("reset_time").get<std::uint32_t>(),
            parameters.at("maturity").get<std::uint32_t>(),
        };
        const std::string prefix =
            "Forward-start option row id '" + row_id + "': ";
        if (!std::isfinite(product.moneyness) || !(product.moneyness > 0.0f))
            throw std::invalid_argument(prefix + "moneyness must be finite and positive.");
        if (product.reset_time == 0U)
            throw std::invalid_argument(prefix + "reset_time must be a positive business-day count.");
        if (product.maturity == 0U)
            throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
        if (!(product.reset_time < product.maturity))
            throw std::invalid_argument(prefix + "reset_time must precede maturity.");
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
