// Up-no-touch policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/up_no_touch/dataset.hpp"

namespace ai_factory::workbench::product {

template<equity::DenseEquitySchedulePolicy Schedule, typename Discount>
using UpNoTouchPricingPolicy = equity::UpTouchPricingPolicy<
    Schedule, Discount, UpNoTouchParameters, false
>;

}  // namespace ai_factory::workbench::product
