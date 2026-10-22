#pragma once

#include "product/best_of_call/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

std::vector<BestOfCallParameters> load_best_of_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
