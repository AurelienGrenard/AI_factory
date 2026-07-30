// Model-independent European-call parameters and dataset loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::products {

// EuropeanCallInput is the compact FP32 product row transferred to CUDA.
struct EuropeanCallInput {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<EuropeanCallInput>);

// Load every European-call row into one contiguous FP32 vector.
std::vector<EuropeanCallInput> load_european_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::products
