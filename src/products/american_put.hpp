// Model-independent American-put parameters and dataset loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::products {

// AmericanPutInput is the compact FP32 product row transferred to CUDA.
struct AmericanPutInput {
    float strike;
    float maturity;
    float exercise_interval;
};

static_assert(std::is_trivially_copyable_v<AmericanPutInput>);

// Load every American-put row into one contiguous FP32 vector.
std::vector<AmericanPutInput> load_american_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::products
