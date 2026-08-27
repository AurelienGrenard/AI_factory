// Down-and-in option policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/down_and_in_option/parameters.hpp"

namespace ai_factory::workbench::product {

template<OptionSide Side>
using DownAndInOptionPathPolicy = equity::SingleBarrierOptionPathPolicy<
    DownAndInOptionParameters,
    Side,
    payoff::BarrierDirection::down,
    true
>;

template<simulation::DenseSchedulePolicy Schedule, OptionSide Side>
using DownAndInOptionPricingPolicy = equity::SingleBarrierOptionPricingPolicy<
    Schedule, DownAndInOptionParameters, Side,
    payoff::BarrierDirection::down, true
>;

}  // namespace ai_factory::workbench::product
