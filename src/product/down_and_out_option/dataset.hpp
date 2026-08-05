// Down-and-out-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DownAndOutOptionParameters is the compact FP32 product row transferred to CUDA.
struct DownAndOutOptionParameters {
    float strike;
    float barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<DownAndOutOptionParameters>);

// Load every Down-and-out-option row into one contiguous FP32 vector.
std::vector<DownAndOutOptionParameters> load_down_and_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
