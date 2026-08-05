// Up-and-in-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// UpAndInOptionParameters is the compact FP32 product row transferred to CUDA.
struct UpAndInOptionParameters {
    float strike;
    float barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<UpAndInOptionParameters>);

// Load every Up-and-in-option row into one contiguous FP32 vector.
std::vector<UpAndInOptionParameters> load_up_and_in_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
