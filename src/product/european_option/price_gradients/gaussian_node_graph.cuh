// European-option binding for the Gaussian-Volterra FFT graph.
#pragma once

#include "common/volterra/price_gradients/gaussian_terminal_product_graph.cuh"
#include "product/european_option/price_gradients/node_graph.cuh"

namespace ai_factory::workbench::product::european_option::price_gradients {

template<
    typename ModelPreparation,
    typename ModelPathPolicy,
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
using GaussianEuropeanOptionNodeGraph =
    vpg::GaussianTerminalProductNodeGraph<
        ModelPreparation, ModelPathPolicy,
        DevicePreparation,
        product::EuropeanOptionGradientPathPolicy<Side>,
        Orders, MaximumSensitivities
    >;

}  // namespace ai_factory::workbench::product::european_option::price_gradients
