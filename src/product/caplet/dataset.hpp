// Caplet dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact FP32 caplet contract transferred to CUDA.
struct CapletParameters {
    float notional;
    float strike;
    float fixing_time;
    float payment_time;
    float accrual_period;
};

static_assert(std::is_trivially_copyable_v<CapletParameters>);

// Load every caplet row into one contiguous FP32 vector.
std::vector<CapletParameters> load_caplets(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
