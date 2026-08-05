// European-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// EuropeanOptionParameters is the compact FP32 product row transferred to CUDA.
struct EuropeanOptionParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<EuropeanOptionParameters>);

// Load every European-option row into one contiguous FP32 vector.
std::vector<EuropeanOptionParameters> load_european_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
