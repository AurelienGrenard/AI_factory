// Convert forward-rate-option JSON rows into compact CUDA parameters.
#include "product/rate_option/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<RateOptionParameters> load_rate_options(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open rate option JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid rate option JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<RateOptionParameters> products;
    products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const RateOptionParameters product = {
            parameters.at("notional").get<float>(),
            parameters.at("strike").get<float>(),
            parameters.at("fixing_time").get<float>(),
            parameters.at("payment_time").get<float>(),
            parameters.at("accrual_period").get<float>(),
        };
        const std::string prefix = "Rate option row id '" + row_id + "': ";
        if (!std::isfinite(product.notional) || !(product.notional > 0.0f))
            throw std::invalid_argument(prefix + "notional must be finite and positive.");
        if (!std::isfinite(product.strike) || !(product.strike >= 0.0f))
            throw std::invalid_argument(prefix + "strike must be finite and non-negative.");
        if (!std::isfinite(product.fixing_time) || !(product.fixing_time > 0.0f))
            throw std::invalid_argument(prefix + "fixing_time must be finite and positive.");
        if (!std::isfinite(product.payment_time)
            || !(product.payment_time > product.fixing_time)) {
            throw std::invalid_argument(
                prefix + "payment_time must be finite and above fixing_time."
            );
        }
        if (!std::isfinite(product.accrual_period)
            || !(product.accrual_period > 0.0f)) {
            throw std::invalid_argument(
                prefix + "accrual_period must be finite and positive."
            );
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
