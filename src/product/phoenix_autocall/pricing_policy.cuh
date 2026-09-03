// Phoenix-autocall policy specialization.
#pragma once

#include "product/phoenix_autocall/parameters.hpp"
#include "product/phoenix_coupon_memory_path_policy.cuh"

namespace ai_factory::workbench::product {

using PhoenixAutocallPathPolicy = detail::PhoenixCouponMemoryPathPolicy<
    PhoenixAutocallParameters,
    false
>;

template<simulation::CountedObservedSchedulePolicy Schedule>
using PhoenixAutocallPricingPolicy = detail::PhoenixCouponMemoryPricingPolicy<
    Schedule, PhoenixAutocallParameters, false
>;

}  // namespace ai_factory::workbench::product
