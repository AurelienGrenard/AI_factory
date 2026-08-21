// Down-and-out-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DownAndOutOptionParameters is the compact product row transferred to CUDA.
struct DownAndOutOptionParameters {
    float strike;
    float barrier;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<DownAndOutOptionParameters>);

// Load every Down-and-out-option row into one contiguous vector.
std::vector<DownAndOutOptionParameters> load_down_and_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
