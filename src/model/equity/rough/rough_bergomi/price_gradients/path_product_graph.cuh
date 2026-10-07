// Path sensitivity graph composition for rough_bergomi products.
#pragma once

#include "common/volterra/price_gradients/gaussian_path_product_graph.cuh"
#include "model/equity/rough/rough_bergomi/price_gradients/terminal_product_graph.cuh"

namespace ai_factory::workbench::model::equity::rough_bergomi {

template<typename ProductPreparation>
using PathProductPriceGradientPlan =
    TerminalProductPriceGradientPlan<ProductPreparation>;

template<typename ProductPreparation>
auto prepare_path_product_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const typename ProductPreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_terminal_product_sensitivities<ProductPreparation>(
        models, products, construction, time, configuration, request
    );
}

template<
    typename ProductPreparation,
    typename ProductPolicy,
    typename HybridSchedule,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using PathProductNodeGraph = vpg::GaussianPathProductNodeGraph<
    price_gradients::DevicePreparation,
    PathPolicy,
    ProductPreparation,
    ProductPolicy,
    HybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
