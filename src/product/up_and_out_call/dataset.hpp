// Up-and-out-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// UpAndOutCallParameters is the compact FP32 product row transferred to CUDA.
struct UpAndOutCallParameters {
    float strike;
    float barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<UpAndOutCallParameters>);

// Load every Up-and-out-call row into one contiguous FP32 vector.
std::vector<UpAndOutCallParameters> load_up_and_out_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
