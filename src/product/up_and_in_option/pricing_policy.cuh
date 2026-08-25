// Up-and-in option policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/up_and_in_option/parameters.hpp"

namespace ai_factory::workbench::product {

template<simulation::DenseSchedulePolicy Schedule, OptionSide Side>
using UpAndInOptionPricingPolicy = equity::SingleBarrierOptionPricingPolicy<
    Schedule, UpAndInOptionParameters, Side,
    payoff::BarrierDirection::up, true
>;

}  // namespace ai_factory::workbench::product
