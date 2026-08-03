// Down-and-in-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DownAndInPutParameters is the compact FP32 product row transferred to CUDA.
struct DownAndInPutParameters {
    float strike;
    float barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<DownAndInPutParameters>);

// Load every Down-and-in-put row into one contiguous FP32 vector.
std::vector<DownAndInPutParameters> load_down_and_in_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
