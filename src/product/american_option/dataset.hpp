// American-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// AmericanOptionParameters is the compact product row transferred to CUDA.
struct AmericanOptionParameters {
    float strike;
    std::uint32_t maturity;
    std::uint32_t exercise_interval;
};

static_assert(std::is_trivially_copyable_v<AmericanOptionParameters>);

// Load every American-option row into one contiguous vector.
std::vector<AmericanOptionParameters> load_american_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
