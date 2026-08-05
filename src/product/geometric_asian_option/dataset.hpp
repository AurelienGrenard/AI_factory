// Geometric-Asian-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// GeometricAsianOptionParameters is the compact FP32 product row transferred to CUDA.
struct GeometricAsianOptionParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<GeometricAsianOptionParameters>);

// Load every Geometric-Asian-option row into one contiguous FP32 vector.
std::vector<GeometricAsianOptionParameters> load_geometric_asian_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
