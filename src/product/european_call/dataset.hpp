// European-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// EuropeanCallParameters is the compact FP32 product row transferred to CUDA.
struct EuropeanCallParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<EuropeanCallParameters>);

// Load every European-call row into one contiguous FP32 vector.
std::vector<EuropeanCallParameters> load_european_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
