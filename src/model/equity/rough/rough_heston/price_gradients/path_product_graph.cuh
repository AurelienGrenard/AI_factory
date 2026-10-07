// Path sensitivity graph composition for rough_heston products.
#pragma once

#include "common/volterra/price_gradients/prepared_lift_path_product_graph.cuh"
#include "model/equity/rough/rough_heston/price_gradients/terminal_product_graph.cuh"

namespace ai_factory::workbench::model::equity::rough_heston {

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
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using PathProductNodeGraph =
    vpg::PreparedLiftPathProductGraph<
        DynamicsPolicy<FactorCount>,
        TerminalProductPriceGradientPlan<ProductPreparation>,
        price_gradients::PreparedLiftNodeCache<FactorCount>,
        ProductPolicy, HybridSchedule, Orders, MaximumSensitivities
    >;

}  // namespace ai_factory::workbench::model::equity::rough_heston
