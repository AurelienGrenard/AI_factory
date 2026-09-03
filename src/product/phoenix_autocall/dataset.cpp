// Convert Phoenix-autocall JSON rows into compact CUDA parameters.
#include "product/phoenix_autocall/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and enforce its issuance schedule and barrier ordering.
std::vector<PhoenixAutocallParameters> load_phoenix_autocalls(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<PhoenixAutocallParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Phoenix autocall",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const PhoenixAutocallParameters product = {
                parameters.at("maturity").get<std::uint32_t>(),
                parameters.at("observation_interval").get<std::uint32_t>(),
                parameters.at("autocall_barrier").get<float>(),
                parameters.at("coupon_barrier").get<float>(),
                parameters.at("protection_barrier").get<float>(),
                parameters.at("annual_coupon_rate").get<float>(),
            };
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            if (product.observation_interval_days == 0U
                || product.observation_interval_days > product.maturity_days) {
                throw std::invalid_argument(
                    prefix
                    + "observation_interval must be positive and at most maturity."
                );
            }
            if (product.maturity_days % product.observation_interval_days != 0U) {
                throw std::invalid_argument(
                    prefix
                    + "maturity must be an integer multiple of observation_interval."
                );
            }
            if (!std::isfinite(product.protection_barrier)
                || !(product.protection_barrier > 0.0f)
                || !std::isfinite(product.coupon_barrier)
                || !std::isfinite(product.autocall_barrier)
                || !(product.protection_barrier <= product.coupon_barrier)
                || !(product.coupon_barrier <= product.autocall_barrier)) {
                throw std::invalid_argument(
                    prefix
                    + "barriers must be finite, positive, and ordered as "
                      "protection <= coupon <= autocall."
                );
            }
            if (!std::isfinite(product.annual_coupon_rate)
                || !(product.annual_coupon_rate > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "annual_coupon_rate must be finite and positive."
                );
            }
            return product;
    });
}

}  // namespace ai_factory::workbench::product
