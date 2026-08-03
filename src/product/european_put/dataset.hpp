// European-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// EuropeanPutParameters is the compact FP32 product row transferred to CUDA.
struct EuropeanPutParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<EuropeanPutParameters>);

// Load every European-put row into one contiguous FP32 vector.
std::vector<EuropeanPutParameters> load_european_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
