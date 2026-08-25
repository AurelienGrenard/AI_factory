// Down-and-out option policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/down_and_out_option/parameters.hpp"

namespace ai_factory::workbench::product {

template<simulation::DenseSchedulePolicy Schedule, OptionSide Side>
using DownAndOutOptionPricingPolicy = equity::SingleBarrierOptionPricingPolicy<
    Schedule, DownAndOutOptionParameters, Side,
    payoff::BarrierDirection::down, false
>;

}  // namespace ai_factory::workbench::product
