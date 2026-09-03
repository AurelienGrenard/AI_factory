// Phoenix-memory-autocall policy specialization.
#pragma once

#include "product/phoenix_autocall/pricing_policy_core.cuh"
#include "product/phoenix_memory_autocall/parameters.hpp"

namespace ai_factory::workbench::product {

using PhoenixMemoryAutocallPathPolicy = detail::PhoenixPathPolicy<
    PhoenixMemoryAutocallParameters,
    true
>;

template<simulation::CountedObservedSchedulePolicy Schedule>
using PhoenixMemoryAutocallPricingPolicy = detail::PhoenixPricingPolicy<
    Schedule, PhoenixMemoryAutocallParameters, true
>;

}  // namespace ai_factory::workbench::product
