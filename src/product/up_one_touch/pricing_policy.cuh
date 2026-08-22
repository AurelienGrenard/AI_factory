// Up-one-touch policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/up_one_touch/dataset.hpp"

namespace ai_factory::workbench::product {

template<equity::DenseEquitySchedulePolicy Schedule, typename Discount>
using UpOneTouchPricingPolicy = equity::UpTouchPricingPolicy<
    Schedule, Discount, UpOneTouchParameters, true
>;

}  // namespace ai_factory::workbench::product
