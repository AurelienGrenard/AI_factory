// Up-and-out option policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/up_and_out_option/parameters.hpp"

namespace ai_factory::workbench::product {

template<OptionSide Side>
using UpAndOutOptionPathPolicy = equity::SingleBarrierOptionPathPolicy<
    UpAndOutOptionParameters,
    Side,
    payoff::BarrierDirection::up,
    false
>;

template<simulation::DenseSchedulePolicy Schedule, OptionSide Side>
using UpAndOutOptionPricingPolicy = equity::SingleBarrierOptionPricingPolicy<
    Schedule, UpAndOutOptionParameters, Side,
    payoff::BarrierDirection::up, false
>;

}  // namespace ai_factory::workbench::product
