// Terminal sensitivity graph composition for rough_sabr products.
#pragma once

#include "common/volterra/price_gradients/gaussian_terminal_product_graph.cuh"
#include "model/equity/rough/rough_sabr/dynamics_impl.cuh"
#include "model/equity/rough/rough_sabr/price_gradients/device_preparation.cuh"

namespace ai_factory::workbench::model::equity::rough_sabr {

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
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using TerminalProductNodeGraph =
    vpg::GaussianTerminalProductNodeGraph<
        price_gradients::DevicePreparation, PathPolicy,
        ProductPreparation, ProductPolicy,
        Orders, MaximumSensitivities
    >;

}  // namespace ai_factory::workbench::model::equity::rough_sabr
