// Geometric-Asian-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// GeometricAsianPutParameters is the compact FP32 product row transferred to CUDA.
struct GeometricAsianPutParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<GeometricAsianPutParameters>);

// Load every Geometric-Asian-put row into one contiguous FP32 vector.
std::vector<GeometricAsianPutParameters> load_geometric_asian_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
