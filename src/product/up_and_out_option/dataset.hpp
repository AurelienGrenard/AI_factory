// Up-and-out-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// UpAndOutOptionParameters is the compact product row transferred to CUDA.
struct UpAndOutOptionParameters {
    float strike;
    float barrier;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<UpAndOutOptionParameters>);

// Load every Up-and-out-option row into one contiguous vector.
std::vector<UpAndOutOptionParameters> load_up_and_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
