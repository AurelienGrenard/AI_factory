// Down-and-in-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DownAndInOptionParameters is the compact FP32 product row transferred to CUDA.
struct DownAndInOptionParameters {
    float strike;
    float barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<DownAndInOptionParameters>);

// Load every Down-and-in-option row into one contiguous FP32 vector.
std::vector<DownAndInOptionParameters> load_down_and_in_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
