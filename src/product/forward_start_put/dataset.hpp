// Forward-start-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// ForwardStartPutParameters is the compact FP32 product row transferred to CUDA.
struct ForwardStartPutParameters {
    float moneyness;
    float reset_time;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<ForwardStartPutParameters>);

// Load every Forward-start-put row into one contiguous FP32 vector.
std::vector<ForwardStartPutParameters> load_forward_start_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
