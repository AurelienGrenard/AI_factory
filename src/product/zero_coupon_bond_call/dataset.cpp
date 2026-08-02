// Convert zero-coupon bond call JSON rows into compact CUDA parameters.
#include "product/zero_coupon_bond_call/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<ZeroCouponBondCallParameters> load_zero_coupon_bond_calls(
    const std::filesystem::path& dataset_path
) {
    std::ifstream stream(dataset_path);
    if (!stream) {
        throw std::runtime_error(
            "Could not open zero-coupon bond call JSON: "
            + dataset_path.string()
        );
    }

    nlohmann::json document;
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Invalid zero-coupon bond call JSON '" + dataset_path.string()
            + "': " + error.what()
        );
    }

    datasets::validate_product_dataset(document);
    const auto& rows = document.at("products");
    std::vector<ZeroCouponBondCallParameters> products;
    products.reserve(rows.size());
    for (const auto& row : rows) {
        const std::string row_id = row.at("id").get<std::string>();
        const auto& parameters = row.at("parameters");
        const ZeroCouponBondCallParameters product = {
            parameters.at("notional").get<float>(),
            parameters.at("strike").get<float>(),
            parameters.at("option_expiry").get<float>(),
            parameters.at("bond_maturity").get<float>(),
        };
        const std::string prefix =
            "Zero-coupon bond call row id '" + row_id + "': ";
        if (!std::isfinite(product.notional) || !(product.notional > 0.0f))
            throw std::invalid_argument(prefix + "notional must be finite and positive.");
        if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
            throw std::invalid_argument(prefix + "strike must be finite and positive.");
        if (!std::isfinite(product.option_expiry)
            || !(product.option_expiry > 0.0f)) {
            throw std::invalid_argument(
                prefix + "option_expiry must be finite and positive."
            );
        }
        if (!std::isfinite(product.bond_maturity)
            || !(product.bond_maturity > product.option_expiry)) {
            throw std::invalid_argument(
                prefix + "bond_maturity must be finite and above option_expiry."
            );
        }
        products.push_back(product);
    }
    return products;
}

}  // namespace ai_factory::workbench::product
