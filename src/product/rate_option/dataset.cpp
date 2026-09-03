// Convert forward-rate-option JSON rows into compact CUDA parameters.
#include "product/rate_option/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<RateOptionParameters> load_rate_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<RateOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Rate option",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const RateOptionParameters product = {
                parameters.at("notional").get<float>(),
                parameters.at("strike").get<float>(),
                parameters.at("fixing_time").get<std::uint32_t>(),
                parameters.at("payment_time").get<std::uint32_t>(),
                parameters.at("accrual_period").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.notional) || !(product.notional > 0.0f))
                throw std::invalid_argument(prefix + "notional must be finite and positive.");
            // Negative strikes are valid for rate options in negative-rate regimes.
            if (!std::isfinite(product.strike))
                throw std::invalid_argument(prefix + "strike must be finite.");
            if (product.fixing_time_days == 0U)
                throw std::invalid_argument(prefix + "fixing_time must be positive.");
            if (!(product.payment_time_days > product.fixing_time_days)) {
                throw std::invalid_argument(
                    prefix + "payment_time must be above fixing_time."
                );
            }
            if (product.accrual_period_days == 0U) {
                throw std::invalid_argument(
                    prefix + "accrual_period must be positive."
                );
            }
            if (product.payment_time_days - product.fixing_time_days
                != product.accrual_period_days) {
                throw std::invalid_argument(
                    prefix + "accrual_period must equal payment_time - fixing_time."
                );
            }
            const float accrual_years =
                static_cast<float>(product.accrual_period_days) / 252.0f;
            if (!(std::fma(
                    accrual_years, product.strike, 1.0f
                ) > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "1 + accrual_period * strike must be positive."
                );
            }
            return product;
    });
}

}  // namespace ai_factory::workbench::product
