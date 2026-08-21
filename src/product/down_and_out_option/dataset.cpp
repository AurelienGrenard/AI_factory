// Convert Down-and-out-option JSON rows into compact CUDA parameters.
#include "product/down_and_out_option/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<DownAndOutOptionParameters> load_down_and_out_options(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Down-and-out option JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Down-and-out option JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<DownAndOutOptionParameters> products;
    products.reserve(rows.size());
    // Only strike and maturity are retained; JSON metadata stays out of CUDA.
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const DownAndOutOptionParameters product = {
            parameters.at("strike").get<float>(),
            parameters.at("barrier").get<float>(),
            parameters.at("maturity").get<std::uint32_t>(),
        };
        const std::string prefix =
            "Down-and-out option row id '" + row_id + "': ";
        if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
            throw std::invalid_argument(prefix + "strike must be finite and positive.");
        if (!std::isfinite(product.barrier) || !(product.barrier > 0.0f))
            throw std::invalid_argument(prefix + "barrier must be finite and positive.");
        if (!(product.barrier < product.strike))
            throw std::invalid_argument(prefix + "barrier must be below strike.");
        if (product.maturity == 0U)
            throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
