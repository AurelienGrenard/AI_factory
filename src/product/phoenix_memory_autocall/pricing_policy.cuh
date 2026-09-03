// Phoenix-memory-autocall policy specialization.
#pragma once

#include "product/phoenix_coupon_memory_path_policy.cuh"
#include "product/phoenix_memory_autocall/parameters.hpp"

namespace ai_factory::workbench::product {

using PhoenixMemoryAutocallPathPolicy = detail::PhoenixCouponMemoryPathPolicy<
    PhoenixMemoryAutocallParameters,
    true
>;

template<simulation::CountedObservedSchedulePolicy Schedule>
using PhoenixMemoryAutocallPricingPolicy =
    detail::PhoenixCouponMemoryPricingPolicy<
    Schedule, PhoenixMemoryAutocallParameters, true
>;

}  // namespace ai_factory::workbench::product
