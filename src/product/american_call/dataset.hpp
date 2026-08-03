// American-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// AmericanCallParameters is the compact FP32 product row transferred to CUDA.
struct AmericanCallParameters {
    float strike;
    float maturity;
    float exercise_interval;
};

static_assert(std::is_trivially_copyable_v<AmericanCallParameters>);

// Load every American-call row into one contiguous FP32 vector.
std::vector<AmericanCallParameters> load_american_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
