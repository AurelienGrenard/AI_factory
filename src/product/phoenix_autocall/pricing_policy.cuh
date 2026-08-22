// Phoenix-autocall policy specialization.
#pragma once

#include "product/phoenix_autocall/dataset.hpp"
#include "product/phoenix_autocall/pricing_policy_core.cuh"

namespace ai_factory::workbench::product {

template<equity::EquitySchedulePolicy Schedule, typename Discount>
using PhoenixAutocallPricingPolicy = detail::PhoenixPricingPolicy<
    Schedule, Discount, PhoenixAutocallParameters, false
>;

}  // namespace ai_factory::workbench::product
