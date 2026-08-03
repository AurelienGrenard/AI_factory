// Double-knock-out-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DoubleKnockOutPutParameters is the compact FP32 product row transferred to CUDA.
struct DoubleKnockOutPutParameters {
    float strike;
    float lower_barrier;
    float upper_barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<DoubleKnockOutPutParameters>);

// Load every Double-knock-out-put row into one contiguous FP32 vector.
std::vector<DoubleKnockOutPutParameters> load_double_knock_out_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
