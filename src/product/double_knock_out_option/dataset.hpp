// Double-knock-out-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DoubleKnockOutOptionParameters is the compact FP32 product row transferred to CUDA.
struct DoubleKnockOutOptionParameters {
    float strike;
    float lower_barrier;
    float upper_barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<DoubleKnockOutOptionParameters>);

// Load every Double-knock-out-option row into one contiguous FP32 vector.
std::vector<DoubleKnockOutOptionParameters> load_double_knock_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
