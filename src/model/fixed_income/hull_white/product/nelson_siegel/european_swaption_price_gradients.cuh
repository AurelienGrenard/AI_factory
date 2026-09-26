// Hull-White/Nelson-Siegel European-swaption sensitivities under Jamshidian.
#pragma once

#include "common/closed_form/concepts.cuh"
#include "common/fixed_income/price_gradients/device_prepared_plan.hpp"
#include "common/fixed_income/swaption_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/fixed_income/hull_white/price_gradients/device_preparation.cuh"
#include "product/european_swaption/price_gradients/device_preparation.cuh"
#include "curve/nelson_siegel/price_gradients/device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace fipg = ::ai_factory::workbench::fixed_income::price_gradients;
namespace model_pg =
    ::ai_factory::workbench::model::fixed_income::hull_white::price_gradients;
namespace european_swaption_pg =
    ::ai_factory::workbench::product::european_swaption::price_gradients;
namespace curve_pg =
    ::ai_factory::workbench::curve::nelson_siegel::price_gradients;
using CurveParameters = curve::nelson_siegel::NelsonSiegelParameters;

using EuropeanSwaptionPriceGradientPlan = fipg::CurveDevicePreparedPlan<
    model_pg::DevicePreparation,
    curve_pg::DevicePreparation,
    european_swaption_pg::DevicePreparation
>;

void prepare_european_swaption_price_gradient_stencils_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_european_swaption_diagonal_sensitivity_stencils_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline EuropeanSwaptionPriceGradientPlan
prepare_hull_white_nelson_siegel_european_swaption_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const curve::nelson_siegel::NelsonSiegelParameters> curves,
    std::span<const product::RegularEuropeanSwaptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return fipg::prepare_curve_device_sensitivities<EuropeanSwaptionPriceGradientPlan>(
        models,
        curves,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline EuropeanSwaptionPriceGradientPlan
prepare_hull_white_nelson_siegel_european_swaption_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const curve::nelson_siegel::NelsonSiegelParameters> curves,
    std::span<const product::RegularEuropeanSwaptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_hull_white_nelson_siegel_european_swaption_sensitivities(
        models,
        curves,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<SwaptionSide Side>
void launch_hull_white_nelson_siegel_european_swaption_price_gradients_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs,
    closed_form::WorkDistribution distribution =
        closed_form::WorkDistribution::cooperative
);

template<SwaptionSide Side, pg::SensitivityOrders Orders>
void launch_hull_white_nelson_siegel_european_swaption_diagonal_sensitivities_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    closed_form::WorkDistribution distribution =
        closed_form::WorkDistribution::cooperative
);

}  // namespace ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel
