// Down-and-out-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// DownAndOutPutParameters is the compact FP32 product row transferred to CUDA.
struct DownAndOutPutParameters {
    float strike;
    float barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<DownAndOutPutParameters>);

// Load every Down-and-out-put row into one contiguous FP32 vector.
std::vector<DownAndOutPutParameters> load_down_and_out_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
