// Zero-coupon bond call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact FP32 zero-coupon bond call transferred to CUDA.
struct ZeroCouponBondCallParameters {
    float notional;
    float strike;
    float option_expiry;
    float bond_maturity;
};

static_assert(std::is_trivially_copyable_v<ZeroCouponBondCallParameters>);

// Load every zero-coupon bond call into one contiguous FP32 vector.
std::vector<ZeroCouponBondCallParameters> load_zero_coupon_bond_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
