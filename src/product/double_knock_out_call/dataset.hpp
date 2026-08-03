// Double-knock-out-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DoubleKnockOutCallParameters is the compact FP32 product row transferred to CUDA.
struct DoubleKnockOutCallParameters {
    float strike;
    float lower_barrier;
    float upper_barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<DoubleKnockOutCallParameters>);

// Load every Double-knock-out-call row into one contiguous FP32 vector.
std::vector<DoubleKnockOutCallParameters> load_double_knock_out_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
