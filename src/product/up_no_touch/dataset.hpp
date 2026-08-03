// Up-no-touch dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// UpNoTouchParameters is the compact FP32 product row transferred to CUDA.
struct UpNoTouchParameters {
    float barrier;
    float cash_payoff;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<UpNoTouchParameters>);

// Load every Up-no-touch row into one contiguous FP32 vector.
std::vector<UpNoTouchParameters> load_up_no_touches(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
