// Up-one-touch policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/up_one_touch/parameters.hpp"

namespace ai_factory::workbench::product {

using UpOneTouchPathPolicy = equity::UpTouchPathPolicy<
    UpOneTouchParameters,
    true
>;

template<simulation::DenseSchedulePolicy Schedule>
using UpOneTouchPricingPolicy = equity::UpTouchPricingPolicy<
    Schedule, UpOneTouchParameters, true
>;

}  // namespace ai_factory::workbench::product
