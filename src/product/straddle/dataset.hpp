// Straddle dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// StraddleParameters is the compact FP32 product row transferred to CUDA.
struct StraddleParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<StraddleParameters>);

// Load every straddle row into one contiguous FP32 vector.
std::vector<StraddleParameters> load_straddles(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
