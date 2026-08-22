// Up-and-in option policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/up_and_in_option/dataset.hpp"

namespace ai_factory::workbench::product {

template<equity::DenseEquitySchedulePolicy Schedule, typename Discount,
         OptionSide Side>
using UpAndInOptionPricingPolicy = equity::SingleBarrierOptionPricingPolicy<
    Schedule, Discount, UpAndInOptionParameters, Side,
    equity::BarrierDirection::up, true
>;

}  // namespace ai_factory::workbench::product
