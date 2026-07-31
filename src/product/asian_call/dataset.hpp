// Asian-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// AsianCallParameters is the compact FP32 product row transferred to CUDA.
struct AsianCallParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<AsianCallParameters>);

// Load every Asian-call row into one contiguous FP32 vector.
std::vector<AsianCallParameters> load_asian_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
