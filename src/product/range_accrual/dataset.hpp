// Range Accrual dataset row and host-side JSON loader.
#pragma once

#include "product/range_accrual/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Range Accrual row into one contiguous vector.
std::vector<RangeAccrualParameters> load_range_accruals(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
