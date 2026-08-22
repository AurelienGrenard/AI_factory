// Phoenix-memory-autocall policy specialization.
#pragma once

#include "product/phoenix_autocall/pricing_policy_core.cuh"
#include "product/phoenix_memory_autocall/dataset.hpp"

namespace ai_factory::workbench::product {

template<equity::EquitySchedulePolicy Schedule, typename Discount>
using PhoenixMemoryAutocallPricingPolicy = detail::PhoenixPricingPolicy<
    Schedule, Discount, PhoenixMemoryAutocallParameters, true
>;

}  // namespace ai_factory::workbench::product
