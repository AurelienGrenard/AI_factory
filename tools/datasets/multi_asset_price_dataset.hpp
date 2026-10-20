#pragma once

#include "common/multi_asset_price_row.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

struct MultiAssetPriceDatasetRecipe {
    std::vector<std::filesystem::path> model_dataset_paths;
    std::filesystem::path product_dataset_path;
    std::filesystem::path correlation_dataset_path;
    std::optional<std::filesystem::path> curve_dataset_path;
    std::filesystem::path dataset_path;
    std::filesystem::path catalog_path;
    std::string url;
    std::optional<std::filesystem::path> rate_model_dataset_path;
};

void write_multi_asset_price_dataset(
    const MultiAssetPriceDatasetRecipe& recipe,
    const std::vector<std::uint32_t>& scenario_model_indices,
    const std::vector<monte_carlo::MultiAssetPriceRow>& rows,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    std::size_t paths_per_price,
    std::uint64_t base_seed,
    const nlohmann::ordered_json& execution,
    double wall_seconds,
    double kernel_seconds,
    const std::vector<std::uint32_t>& scenario_rate_indices = {}
);

}  // namespace ai_factory::workbench::datasets
