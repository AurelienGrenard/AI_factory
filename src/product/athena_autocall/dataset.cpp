// Convert Athena-autocall JSON rows into compact CUDA parameters.
#include "product/athena_autocall/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and enforce its issuance schedule and barrier ordering.
std::vector<AthenaAutocallParameters> load_athena_autocalls(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Athena autocall JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Athena autocall JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<AthenaAutocallParameters> products;
    products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const AthenaAutocallParameters product = {
            parameters.at("maturity").get<std::uint32_t>(),
            parameters.at("observation_interval").get<std::uint32_t>(),
            parameters.at("autocall_barrier").get<float>(),
            parameters.at("protection_barrier").get<float>(),
            parameters.at("annual_coupon_rate").get<float>(),
        };
        const std::string prefix =
            "Athena autocall row id '" + row_id + "': ";
        if (product.maturity == 0U)
            throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
        if (product.observation_interval == 0U
            || product.observation_interval > product.maturity) {
            throw std::invalid_argument(
                prefix
                + "observation_interval must be positive and at most maturity."
            );
        }
        if (product.maturity % product.observation_interval != 0U) {
            throw std::invalid_argument(
                prefix
                + "maturity must be an integer multiple of observation_interval."
            );
        }
        if (!std::isfinite(product.protection_barrier)
            || !(product.protection_barrier > 0.0f)
            || !std::isfinite(product.autocall_barrier)
            || !(product.protection_barrier <= product.autocall_barrier)) {
            throw std::invalid_argument(
                prefix
                + "barriers must be finite, positive, and ordered as "
                  "protection <= autocall."
            );
        }
        if (!std::isfinite(product.annual_coupon_rate)
            || !(product.annual_coupon_rate > 0.0f)) {
            throw std::invalid_argument(
                prefix + "annual_coupon_rate must be finite and positive."
            );
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
