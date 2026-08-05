// American-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// AmericanOptionParameters is the compact FP32 product row transferred to CUDA.
struct AmericanOptionParameters {
    float strike;
    float maturity;
    float exercise_interval;
};

static_assert(std::is_trivially_copyable_v<AmericanOptionParameters>);

// Load every American-option row into one contiguous FP32 vector.
std::vector<AmericanOptionParameters> load_american_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
