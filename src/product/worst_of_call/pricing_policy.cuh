#pragma once

#include "product/terminal_basket_call_path_policy.cuh"
#include "product/worst_of_call/parameters.hpp"

namespace ai_factory::workbench::product {

using WorstOfCallPathPolicy = detail::TerminalBasketCallPathPolicy<
    WorstOfCallParameters, equity::BasketSpotAggregation::Minimum>;

}  // namespace ai_factory::workbench::product
