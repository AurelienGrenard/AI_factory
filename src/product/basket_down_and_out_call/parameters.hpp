// Down-and-out call on the arithmetic mean of initial-spot-normalized assets.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct BasketDownAndOutCallParameters {
    float strike;
    float lower_barrier;
    std::uint32_t maturity_days;
    std::uint32_t monitoring_interval_days;
};

static_assert(std::is_trivially_copyable_v<BasketDownAndOutCallParameters>);

}  // namespace ai_factory::workbench::product
