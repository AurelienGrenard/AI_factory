// Zero-coupon bond call dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact zero-coupon bond option transferred to CUDA.
struct ZeroCouponBondOptionParameters {
    float notional;
    float strike;
    std::uint32_t option_expiry;
    std::uint32_t bond_maturity;
};

static_assert(std::is_trivially_copyable_v<ZeroCouponBondOptionParameters>);

// Load every zero-coupon bond option into one contiguous vector.
std::vector<ZeroCouponBondOptionParameters> load_zero_coupon_bond_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
