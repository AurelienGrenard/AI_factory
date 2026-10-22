// Convert zero-coupon bond option JSON rows into compact CUDA parameters.
#include "product/zero_coupon_bond_option/dataset.hpp"
#include "datasets/validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<ZeroCouponBondOptionParameters> load_zero_coupon_bond_options(
    const std::filesystem::path& dataset_path
) {
    const bool quoted = datasets::fixed_income_relative_quote(
        datasets::read_parameter_dataset(
            dataset_path, datasets::ParameterDatasetFamily::Product,
            "Fixed-income product"),
        "ratio_to_initial_forward_bond_price", "dimensionless");
    return datasets::load_parameter_rows<ZeroCouponBondOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Zero-coupon bond call",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ZeroCouponBondOptionParameters product = {
                parameters.at("notional").get<float>(),
                parameters.at("strike").get<float>(),
                parameters.at("option_expiry").get<std::uint32_t>(),
                parameters.at("bond_maturity").get<std::uint32_t>(),
                quoted,
            };
            if (!std::isfinite(product.notional) || !(product.notional > 0.0f))
                throw std::invalid_argument(prefix + "notional must be finite and positive.");
            if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
                throw std::invalid_argument(prefix + "strike must be finite and positive.");
            if (product.option_expiry_days == 0U) {
                throw std::invalid_argument(
                    prefix + "option_expiry must be positive."
                );
            }
            if (!(product.bond_maturity_days > product.option_expiry_days)) {
                throw std::invalid_argument(
                    prefix + "bond_maturity must be above option_expiry."
                );
            }
            return product;
    });
}

}  // namespace ai_factory::workbench::product
