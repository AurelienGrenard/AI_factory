// Up-and-out option policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/up_and_out_option/dataset.hpp"

namespace ai_factory::workbench::product {

template<equity::DenseEquitySchedulePolicy Schedule, typename Discount,
         OptionSide Side>
using UpAndOutOptionPricingPolicy = equity::SingleBarrierOptionPricingPolicy<
    Schedule, Discount, UpAndOutOptionParameters, Side,
    equity::BarrierDirection::up, false
>;

}  // namespace ai_factory::workbench::product
