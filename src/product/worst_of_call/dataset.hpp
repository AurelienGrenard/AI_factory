#pragma once

#include "product/worst_of_call/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

std::vector<WorstOfCallParameters> load_worst_of_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
