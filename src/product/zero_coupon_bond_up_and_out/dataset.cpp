#include "product/zero_coupon_bond_up_and_out/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

std::vector<ZeroCouponBondUpAndOutParameters> load_zero_coupon_bond_up_and_outs(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<ZeroCouponBondUpAndOutParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Zero-coupon bond up-and-out call",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const ZeroCouponBondUpAndOutParameters product{
                parameters.at("notional").get<float>(),
                parameters.at("strike").get<float>(),
                parameters.at("barrier").get<float>(),
                parameters.at("option_expiry").get<std::uint32_t>(),
                parameters.at("bond_maturity").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.notional) || product.notional <= 0.0f
                || !std::isfinite(product.strike) || product.strike <= 0.0f
                || !std::isfinite(product.barrier) || product.barrier <= 0.0f
                || product.option_expiry_days == 0U
                || product.bond_maturity_days <= product.option_expiry_days) {
                throw std::invalid_argument(prefix + "requires N,K,B > 0 and 0 < T < U.");
            }
            return product;
        }
    );
}

}  // namespace ai_factory::workbench::product
