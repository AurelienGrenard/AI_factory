// Convert asset-or-nothing JSON rows into compact CUDA parameters.
#include "product/asset_or_nothing_option/dataset.hpp"
#include "common/dataset_validation.hpp"
#include "product/asset_or_nothing_option/parameter_domain.hpp"

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
            asset_or_nothing_option::validate_parameters(product, prefix);
            return product;
    });
}

}  // namespace ai_factory::workbench::product
