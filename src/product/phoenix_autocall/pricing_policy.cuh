// Phoenix-autocall policy specialization.
#pragma once

#include "product/phoenix_autocall/parameters.hpp"
#include "product/phoenix_autocall/pricing_policy_core.cuh"

namespace ai_factory::workbench::product {

using PhoenixAutocallPathPolicy = detail::PhoenixPathPolicy<
    PhoenixAutocallParameters,
    false
>;

template<simulation::CountedObservedSchedulePolicy Schedule>
using PhoenixAutocallPricingPolicy = detail::PhoenixPricingPolicy<
    Schedule, PhoenixAutocallParameters, false
>;

}  // namespace ai_factory::workbench::product
