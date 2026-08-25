// Up-no-touch policy specialization.
#pragma once

#include "common/equity/barrier_pricing_policy.cuh"
#include "product/up_no_touch/parameters.hpp"

namespace ai_factory::workbench::product {

template<simulation::DenseSchedulePolicy Schedule>
using UpNoTouchPricingPolicy = equity::UpTouchPricingPolicy<
    Schedule, UpNoTouchParameters, false
>;

}  // namespace ai_factory::workbench::product
