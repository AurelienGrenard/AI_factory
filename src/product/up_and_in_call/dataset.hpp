// Up-and-in-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// UpAndInCallParameters is the compact FP32 product row transferred to CUDA.
struct UpAndInCallParameters {
    float strike;
    float barrier;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<UpAndInCallParameters>);

// Load every Up-and-in-call row into one contiguous FP32 vector.
std::vector<UpAndInCallParameters> load_up_and_in_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
