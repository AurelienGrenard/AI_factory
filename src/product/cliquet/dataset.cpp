// Convert Cliquet JSON rows into compact CUDA parameters.
#include "product/cliquet/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and enforce its schedule and cap-floor ordering.
std::vector<CliquetParameters> load_cliquets(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open Cliquet JSON: " + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid Cliquet JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<CliquetParameters> products;
    products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const CliquetParameters product = {
            parameters.at("maturity").get<float>(),
            parameters.at("observation_interval").get<float>(),
            parameters.at("participation_rate").get<float>(),
            parameters.at("local_floor").get<float>(),
            parameters.at("local_cap").get<float>(),
            parameters.at("global_floor").get<float>(),
            parameters.at("global_cap").get<float>(),
        };
        const std::string prefix = "Cliquet row id '" + row_id + "': ";
        if (!std::isfinite(product.maturity) || !(product.maturity > 0.0f))
            throw std::invalid_argument(
                prefix + "maturity must be finite and positive."
            );
        if (!std::isfinite(product.observation_interval)
            || !(product.observation_interval > 0.0f)
            || product.observation_interval > product.maturity) {
            throw std::invalid_argument(
                prefix + "observation_interval must be finite, positive, "
                "and at most maturity."
            );
        }
        const float raw_count =
            product.maturity / product.observation_interval;
        const float nearest_count = roundf(raw_count);
        const float tolerance =
            32.0f * std::numeric_limits<float>::epsilon()
            * std::max(raw_count, 1.0f);
        if (fabsf(raw_count - nearest_count) > tolerance) {
            throw std::invalid_argument(
                prefix + "maturity must be an integer multiple of "
                "observation_interval."
            );
        }
        if (!std::isfinite(product.participation_rate)
            || !(product.participation_rate > 0.0f)) {
            throw std::invalid_argument(
                prefix + "participation_rate must be finite and positive."
            );
        }
        if (!std::isfinite(product.local_floor)
            || !std::isfinite(product.local_cap)
            || !(product.local_floor < product.local_cap)) {
            throw std::invalid_argument(
                prefix + "local_floor must be finite and below local_cap."
            );
        }
        if (!std::isfinite(product.global_floor)
            || !std::isfinite(product.global_cap)
            || !(product.global_floor > -1.0f)
            || !(product.global_floor < product.global_cap)) {
            throw std::invalid_argument(
                prefix + "global bounds must be finite and satisfy "
                "-1 < global_floor < global_cap."
            );
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
