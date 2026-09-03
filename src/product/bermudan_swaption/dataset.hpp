// Bermudan-swaption dataset loader.
#pragma once

#include "product/bermudan_swaption/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

std::vector<BermudanSwaptionParameters> load_bermudan_swaptions(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
