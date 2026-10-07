// log_modulated_rough_bergomi European-option sensitivities through the Gaussian rough graph.
#pragma once

#include "product/european_option/price_gradients/gaussian_node_graph.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dynamics_impl.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/price_gradients/device_preparation.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace eopg = ::ai_factory::workbench::product::european_option::price_gradients;

using EuropeanOptionPriceGradientPlan =
    eopg::EuropeanOptionPriceGradientPlan<
        price_gradients::DevicePreparation
    >;

inline EuropeanOptionPriceGradientPlan
prepare_log_modulated_rough_bergomi_european_option_sensitivities(
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
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using EuropeanOptionNodeGraph =
    eopg::GaussianEuropeanOptionNodeGraph<
        price_gradients::DevicePreparation,
        PathPolicy, Side, Orders, MaximumSensitivities
    >;

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
