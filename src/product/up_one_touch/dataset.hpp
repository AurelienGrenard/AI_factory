// Up-one-touch dataset row and host-side JSON loader.
#pragma once

#include "product/up_one_touch/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Up-one-touch row into one contiguous vector.
std::vector<UpOneTouchParameters> load_up_one_touches(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
