// Down-and-in option policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/down_and_in_option/dataset.hpp"

namespace ai_factory::workbench::product {

template<equity::DenseEquitySchedulePolicy Schedule, typename Discount,
         OptionSide Side>
using DownAndInOptionPricingPolicy = equity::SingleBarrierOptionPricingPolicy<
    Schedule, Discount, DownAndInOptionParameters, Side,
    equity::BarrierDirection::down, true
>;

}  // namespace ai_factory::workbench::product
