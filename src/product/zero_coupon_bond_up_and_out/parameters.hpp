// Unit-bond call extinguished when the underlying bond reaches its upper barrier.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct ZeroCouponBondUpAndOutParameters {
    float notional;
    float strike;
    float barrier;
    std::uint32_t option_expiry_days;
    std::uint32_t bond_maturity_days;
};

static_assert(std::is_trivially_copyable_v<ZeroCouponBondUpAndOutParameters>);

}  // namespace ai_factory::workbench::product
