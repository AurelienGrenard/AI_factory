// Convert asset-or-nothing JSON rows into compact CUDA parameters.
#include "product/asset_or_nothing_option/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

// Parse one dataset and preserve its row order in the returned vector.
std::vector<AssetOrNothingOptionParameters> load_asset_or_nothing_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<AssetOrNothingOptionParameters>(
        dataset_path,
        datasets::ParameterDatasetFamily::Product,
        "Asset-or-nothing",
        [&](const nlohmann::json& parameters, const std::string& prefix) {
            const AssetOrNothingOptionParameters product = {
                parameters.at("strike").get<float>(),
                parameters.at("maturity").get<std::uint32_t>(),
            };
            if (!std::isfinite(product.strike) || !(product.strike > 0.0f))
                throw std::invalid_argument(prefix + "strike must be finite and positive.");
            if (product.maturity_days == 0U)
                throw std::invalid_argument(prefix + "maturity must be a positive business-day count.");
            return product;
    });
}

}  // namespace ai_factory::workbench::product
