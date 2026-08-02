// Zero-coupon bond put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact FP32 zero-coupon bond put transferred to CUDA.
struct ZeroCouponBondPutParameters {
    float notional;
    float strike;
    float option_expiry;
    float bond_maturity;
};

static_assert(std::is_trivially_copyable_v<ZeroCouponBondPutParameters>);

// Load every zero-coupon bond put into one contiguous FP32 vector.
std::vector<ZeroCouponBondPutParameters> load_zero_coupon_bond_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
