// Double-knock-out-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DoubleKnockOutOptionParameters is the compact product row transferred to CUDA.
struct DoubleKnockOutOptionParameters {
    float strike;
    float lower_barrier;
    float upper_barrier;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<DoubleKnockOutOptionParameters>);

// Load every Double-knock-out-option row into one contiguous vector.
std::vector<DoubleKnockOutOptionParameters> load_double_knock_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
