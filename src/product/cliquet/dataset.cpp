// Convert Cliquet JSON rows into compact CUDA parameters.
#include "product/cliquet/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and enforce its schedule and cap-floor ordering.
std::vector<CliquetParameters> load_cliquets(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<CliquetParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Cliquet",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const CliquetParameters product = {
                parameters.at("maturity").get<std::uint32_t>(),
                parameters.at("observation_interval").get<std::uint32_t>(),
                parameters.at("participation_rate").get<float>(),
                parameters.at("local_floor").get<float>(),
                parameters.at("local_cap").get<float>(),
                parameters.at("global_floor").get<float>(),
                parameters.at("global_cap").get<float>(),
            };
            if (product.maturity_days == 0U)
                throw std::invalid_argument(
                    prefix + "maturity must be a positive business-day count."
                );
            if (product.observation_interval_days == 0U
                || product.observation_interval_days > product.maturity_days) {
                throw std::invalid_argument(
                    prefix + "observation_interval must be positive "
                    "and at most maturity."
                );
            }
            if (product.maturity_days % product.observation_interval_days != 0U) {
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
            return product;
    });
}

}  // namespace ai_factory::workbench::product
