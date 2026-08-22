// Down-and-out option policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/down_and_out_option/dataset.hpp"

namespace ai_factory::workbench::product {

template<equity::DenseEquitySchedulePolicy Schedule, typename Discount,
         OptionSide Side>
using DownAndOutOptionPricingPolicy = equity::SingleBarrierOptionPricingPolicy<
    Schedule, Discount, DownAndOutOptionParameters, Side,
    equity::BarrierDirection::down, false
>;

}  // namespace ai_factory::workbench::product
