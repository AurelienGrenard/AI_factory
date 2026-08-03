// Asian-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// AsianPutParameters is the compact FP32 product row transferred to CUDA.
struct AsianPutParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<AsianPutParameters>);

// Load every Asian-put row into one contiguous FP32 vector.
std::vector<AsianPutParameters> load_asian_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
