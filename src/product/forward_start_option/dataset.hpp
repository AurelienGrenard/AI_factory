// Forward-start-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// ForwardStartOptionParameters is the compact FP32 product row transferred to CUDA.
struct ForwardStartOptionParameters {
    float moneyness;
    float reset_time;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<ForwardStartOptionParameters>);

// Load every Forward-start-option row into one contiguous FP32 vector.
std::vector<ForwardStartOptionParameters> load_forward_start_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
