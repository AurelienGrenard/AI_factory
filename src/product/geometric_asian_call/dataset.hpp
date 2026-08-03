// Geometric-Asian-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// GeometricAsianCallParameters is the compact FP32 product row transferred to CUDA.
struct GeometricAsianCallParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<GeometricAsianCallParameters>);

// Load every Geometric-Asian-call row into one contiguous FP32 vector.
std::vector<GeometricAsianCallParameters> load_geometric_asian_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
