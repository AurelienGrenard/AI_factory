#pragma once

#include "product/best_of_call/parameters.hpp"
#include "product/terminal_basket_call_path_policy.cuh"

namespace ai_factory::workbench::product {

using BestOfCallPathPolicy = detail::TerminalBasketCallPathPolicy<
    BestOfCallParameters, equity::BasketSpotAggregation::Maximum>;

}  // namespace ai_factory::workbench::product
