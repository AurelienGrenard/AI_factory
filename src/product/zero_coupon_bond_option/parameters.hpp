// Zero-coupon-bond-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct ZeroCouponBondOptionParameters {
    float notional;
    float strike;
    std::uint32_t option_expiry_days;
    std::uint32_t bond_maturity_days;
};

static_assert(std::is_trivially_copyable_v<ZeroCouponBondOptionParameters>);

}  // namespace ai_factory::workbench::product
