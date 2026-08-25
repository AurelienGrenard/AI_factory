// Up-no-touch dataset row and host-side JSON loader.
#pragma once

#include "product/up_no_touch/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Up-no-touch row into one contiguous vector.
std::vector<UpNoTouchParameters> load_up_no_touches(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
