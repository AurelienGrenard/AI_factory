// Forward-start-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// ForwardStartCallParameters is the compact FP32 product row transferred to CUDA.
struct ForwardStartCallParameters {
    float moneyness;
    float reset_time;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<ForwardStartCallParameters>);

// Load every Forward-start-call row into one contiguous FP32 vector.
std::vector<ForwardStartCallParameters> load_forward_start_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
