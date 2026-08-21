// Down-and-in-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DownAndInOptionParameters is the compact product row transferred to CUDA.
struct DownAndInOptionParameters {
    float strike;
    float barrier;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<DownAndInOptionParameters>);

// Load every Down-and-in-option row into one contiguous vector.
std::vector<DownAndInOptionParameters> load_down_and_in_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
