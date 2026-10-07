// rough_heston European-option sensitivities through the rough node graph.
#pragma once

#include "product/european_option/price_gradients/node_graph.cuh"
#include "model/equity/rough/rough_heston/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/price_gradients/device_preparation.cuh"
#include "model/equity/rough/rough_heston/price_gradients/lift_node_preparation.hpp"

namespace ai_factory::workbench::model::equity::rough_heston {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace eopg = ::ai_factory::workbench::product::european_option::price_gradients;

using EuropeanOptionPriceGradientPlan =
    eopg::EuropeanOptionPriceGradientPlan<
        price_gradients::DevicePreparation
    >;

inline EuropeanOptionPriceGradientPlan
prepare_rough_heston_european_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::EuropeanOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return eopg::prepare_european_option_sensitivities<
        price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using EuropeanOptionNodeGraph = eopg::EuropeanOptionNodeGraph<
    DynamicsPolicy<FactorCount>,
    EuropeanOptionPriceGradientPlan,
    price_gradients::PreparedLiftNodeCache<FactorCount>,
    Side, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_heston
