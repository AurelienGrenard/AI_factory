// Terminal sensitivity graph composition for rough_heston products.
#pragma once

#include "common/volterra/price_gradients/terminal_product_graph.cuh"
#include "model/equity/rough/rough_heston/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/price_gradients/device_preparation.cuh"
#include "model/equity/rough/rough_heston/price_gradients/lift_node_preparation.hpp"

namespace ai_factory::workbench::model::equity::rough_heston {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace vpg = ::ai_factory::workbench::volterra::price_gradients;

template<typename ProductPreparation>
using TerminalProductPriceGradientPlan =
    vpg::TerminalProductPriceGradientPlan<
        price_gradients::DevicePreparation, ProductPreparation
    >;

template<typename ProductPreparation>
auto prepare_terminal_product_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const typename ProductPreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return vpg::prepare_terminal_product_sensitivities<
        price_gradients::DevicePreparation, ProductPreparation
    >(models, products, construction, time, configuration, request);
}

template<
    typename ProductPreparation,
    typename ProductPolicy,
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using TerminalProductNodeGraph =
    vpg::PreparedLiftTerminalProductGraph<
        DynamicsPolicy<FactorCount>,
        TerminalProductPriceGradientPlan<ProductPreparation>,
        price_gradients::PreparedLiftNodeCache<FactorCount>,
        ProductPolicy, Orders, MaximumSensitivities
    >;

}  // namespace ai_factory::workbench::model::equity::rough_heston
