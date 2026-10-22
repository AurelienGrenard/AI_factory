#pragma once

#include "product/basket_call/parameters.hpp"
#include "product/terminal_basket_call_path_policy.cuh"

namespace ai_factory::workbench::product {

using BasketCallPathPolicy = detail::TerminalBasketCallPathPolicy<
    BasketCallParameters, equity::BasketSpotAggregation::ArithmeticMean>;

}  // namespace ai_factory::workbench::product
