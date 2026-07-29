// Model-independent American-put parameters and registry loader.
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

// Load and convert every row from an American-put registry JSON database.
std::vector<AmericanPutInput> load_american_puts(
    const std::filesystem::path& json_path
);

}  // namespace ai_factory::workbench::products
