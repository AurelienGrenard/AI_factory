// Floorlet dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact FP32 floorlet contract transferred to CUDA.
struct FloorletParameters {
    float notional;
    float strike;
    float fixing_time;
    float payment_time;
    float accrual_period;
};

static_assert(std::is_trivially_copyable_v<FloorletParameters>);

// Load every floorlet row into one contiguous FP32 vector.
std::vector<FloorletParameters> load_floorlets(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
