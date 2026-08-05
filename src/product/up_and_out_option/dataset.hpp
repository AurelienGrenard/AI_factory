// Up-and-out-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// UpAndOutOptionParameters is the compact FP32 product row transferred to CUDA.
struct UpAndOutOptionParameters {
    float strike;
    float barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<UpAndOutOptionParameters>);

// Load every Up-and-out-option row into one contiguous FP32 vector.
std::vector<UpAndOutOptionParameters> load_up_and_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
