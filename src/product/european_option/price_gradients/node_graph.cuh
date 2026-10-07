// European-option binding for the shared prepared-lift node graph.
#pragma once

#include "common/volterra/price_gradients/terminal_product_graph.cuh"
#include "common/option_side.cuh"
#include "product/european_option/parameters.hpp"
#include "product/european_option/price_gradients/device_preparation.cuh"
#include "product/european_option/price_gradients/monte_carlo_policy.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::product::european_option::price_gradients {

namespace vpg = ::ai_factory::workbench::volterra::price_gradients;
namespace pg = ::ai_factory::workbench::price_gradients;

namespace epg = ::ai_factory::workbench::equity::price_gradients;

template<typename ModelPreparation>
using EuropeanOptionPriceGradientPlan =
    vpg::TerminalProductPriceGradientPlan<
        ModelPreparation, DevicePreparation
    >;

template<typename ModelPreparation>
auto prepare_european_option_sensitivities(
    std::span<const typename ModelPreparation::Model> models,
    std::span<const product::EuropeanOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return vpg::prepare_terminal_product_sensitivities<
        ModelPreparation, DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    typename Dynamics,
    typename HostPlan,
    typename Cache,
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
using EuropeanOptionNodeGraph = vpg::PreparedLiftTerminalProductGraph<
    Dynamics, HostPlan, Cache,
    product::EuropeanOptionGradientPathPolicy<Side>,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::product::european_option::price_gradients
