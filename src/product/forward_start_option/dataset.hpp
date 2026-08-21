// Forward-start-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// ForwardStartOptionParameters is the compact product row transferred to CUDA.
struct ForwardStartOptionParameters {
    float moneyness;
    std::uint32_t reset_time;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<ForwardStartOptionParameters>);

// Load every Forward-start-option row into one contiguous vector.
std::vector<ForwardStartOptionParameters> load_forward_start_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
