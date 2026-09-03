// Convert Range Accrual JSON rows into compact CUDA parameters.
#include "product/range_accrual/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and enforce its schedule and observation band.
std::vector<RangeAccrualParameters> load_range_accruals(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<RangeAccrualParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Range Accrual",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const RangeAccrualParameters product = {
                parameters.at("maturity").get<std::uint32_t>(),
                parameters.at("observation_interval").get<std::uint32_t>(),
                parameters.at("lower_barrier").get<float>(),
                parameters.at("upper_barrier").get<float>(),
                parameters.at("coupon_rate").get<float>(),
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
            if (!std::isfinite(product.lower_barrier)
                || !std::isfinite(product.upper_barrier)
                || !(product.lower_barrier > 0.0f)
                || !(product.lower_barrier < 1.0f)
                || !(product.upper_barrier > 1.0f)) {
                throw std::invalid_argument(
                    prefix + "barriers must be finite and satisfy "
                    "0 < lower_barrier < 1 < upper_barrier."
                );
            }
            if (!std::isfinite(product.coupon_rate)
                || !(product.coupon_rate > 0.0f)) {
                throw std::invalid_argument(
                    prefix + "coupon_rate must be finite and positive."
                );
            }
            return product;
    });
}

}  // namespace ai_factory::workbench::product
