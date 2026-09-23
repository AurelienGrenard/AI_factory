// CIR Jamshidian sensitivities: compact preparation, scalar/cooperative parity.
#include "model/fixed_income/cir/product/european_swaption.cuh"
#include "model/fixed_income/cir/product/european_swaption_price_gradients.cuh"
#include "tests/price_gradients/diagonal_cuda_test_support.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

using namespace ai_factory::workbench;
using namespace price_gradient_test;
namespace cir = model::fixed_income::cir;
namespace pg = price_gradients;

const std::vector<cir::ModelParameters> models{
    {{0.60f, 0.040f, 0.150f}, 0.030f},
    {{0.10f, 0.030f, 0.010f}, 0.020f},
    {{0.30f, 0.050f, 0.080f}, 0.000f},
};

const std::vector<product::RegularEuropeanSwaptionParameters> products{
    {1.0f, 0.035f, 0.5f, 252U, 126U, 4U},
    {2.0f, 0.025f, 0.25f, 504U, 63U, 12U},
    {0.7f, 0.045f, 1.0f, 126U, 252U, 3U},
};

const pg::PriceGradientConfiguration full{{
    {"model.mean_reversion", {0.005}},
    {"model.long_term_mean", {0.005}},
    {"model.volatility", {0.005}},
    {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
    {"product.notional", {0.005}},
    {"product.strike", {0.0005, pg::BumpScale::absolute}},
    {"product.accrual_fraction", {0.005}},
}};

auto prepare_first(const pg::PriceGradientConfiguration& selection) {
    return cir::prepare_cir_european_swaption_price_gradients(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection
    );
}

template<pg::SensitivityOrders Orders>
auto prepare_diagonal(const pg::PriceGradientConfiguration& selection) {
    return cir::prepare_cir_european_swaption_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        {Orders}
    );
}

template<SwaptionSide Side>
void check_side() {
    pg::LaunchConfiguration launch{
        pg::PricingMethod::closed_form,
        0U,
        models.size(),
        0U,
        128U,
        2U,
        0U,
    };
    const auto first_launcher = [](
        closed_form::WorkDistribution distribution,
        const auto& plan,
        auto inputs,
        auto stencils,
        const auto& configuration,
        auto outputs
    ) {
        cir::launch_cir_european_swaption_price_gradients_cuda<Side>(
            plan,
            inputs,
            stencils,
            configuration,
            outputs,
            distribution
        );
    };
    const auto scalar_launcher = [&](const auto&... arguments) {
        first_launcher(
            closed_form::WorkDistribution::scalar, arguments...
        );
    };
    const auto cooperative_launcher = [&](const auto&... arguments) {
        first_launcher(
            closed_form::WorkDistribution::cooperative, arguments...
        );
    };

    const auto first_plan = prepare_first(full);
    const auto scalar = execute(first_plan, launch, scalar_launcher, true);
    const auto cooperative = execute(
        first_plan, launch, cooperative_launcher, true
    );
    bool gradients_match = true;
    for (std::size_t row = 0U; row < models.size(); ++row) {
        const float price_scale = std::max(1.0f, std::abs(scalar.price[row]));
        require(
            std::abs(scalar.price[row] - cooperative.price[row])
                <= 3.0e-5f * price_scale,
            "CIR scalar/cooperative central price mismatch."
        );
        for (std::size_t sensitivity = 0U;
             sensitivity < full.sensitivities.size();
             ++sensitivity) {
            const std::size_t index = row * full.sensitivities.size()
                + sensitivity;
            const float scale = std::max(
                1.0f, std::abs(scalar.gradient[index])
            );
            if (std::abs(
                    scalar.gradient[index] - cooperative.gradient[index]
                ) > 2.0e-3f * scale) {
                gradients_match = false;
            }
        }
    }
    require(gradients_match, "CIR scalar/cooperative gradient mismatch.");

    const auto empty = execute(
        prepare_first({}), launch, cooperative_launcher
    );
    const pg::PriceGradientConfiguration reordered{{
        full.sensitivities[5],
        full.sensitivities[3],
        full.sensitivities[0],
    }};
    const auto subset = execute(
        prepare_first(reordered), launch, cooperative_launcher
    );
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(
            empty.price[row],
            cooperative.price[row],
            "CIR selected sensitivities changed the central price"
        );
        for (std::size_t coordinate = 0U; coordinate < 3U; ++coordinate) {
            const auto source = coordinate == 0U ? 5U
                : coordinate == 1U ? 3U : 0U;
            same(
                subset.gradient[row * 3U + coordinate],
                cooperative.gradient[
                    row * full.sensitivities.size() + source
                ],
                "CIR reordered selection changed one gradient"
            );
        }
    }
    require(
        sensitivity_task<pg::SensitivityOrders::first>(
            first_plan, 2U, 3U
        ).stencil.kind == pg::StencilKind::forward,
        "CIR zero initial state did not select a forward stencil."
    );

    const auto diagonal_launcher = [](
        closed_form::WorkDistribution distribution,
        const auto& plan,
        auto inputs,
        auto stencils,
        const auto& configuration,
        auto outputs
    ) {
        cir::launch_cir_european_swaption_diagonal_sensitivities_cuda<
            Side,
            pg::SensitivityOrders::first_and_second
        >(
            plan,
            inputs,
            stencils,
            configuration,
            outputs,
            distribution
        );
    };
    const auto scalar_diagonal_launcher = [&](const auto&... arguments) {
        diagonal_launcher(
            closed_form::WorkDistribution::scalar, arguments...
        );
    };
    const auto cooperative_diagonal_launcher = [&](const auto&... arguments) {
        diagonal_launcher(
            closed_form::WorkDistribution::cooperative, arguments...
        );
    };
    const auto diagonal_plan = prepare_diagonal<
        pg::SensitivityOrders::first_and_second
    >(full);
    const auto scalar_diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(diagonal_plan, launch, scalar_diagonal_launcher);
    const auto cooperative_diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(diagonal_plan, launch, cooperative_diagonal_launcher);

    const auto second_only_launcher = [](
        const auto& plan,
        auto inputs,
        auto stencils,
        const auto& configuration,
        auto outputs
    ) {
        cir::launch_cir_european_swaption_diagonal_sensitivities_cuda<
            Side,
            pg::SensitivityOrders::second
        >(
            plan,
            inputs,
            stencils,
            configuration,
            outputs,
            closed_form::WorkDistribution::cooperative
        );
    };
    const auto second_only = execute_diagonal<
        pg::SensitivityOrders::second
    >(
        prepare_diagonal<pg::SensitivityOrders::second>(full),
        launch,
        second_only_launcher
    );

    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(
            cooperative.price[row],
            cooperative_diagonal.price[row],
            "CIR diagonal request changed the central price bits"
        );
        for (std::size_t sensitivity = 0U;
             sensitivity < full.sensitivities.size();
             ++sensitivity) {
            const std::size_t index = row * full.sensitivities.size()
                + sensitivity;
            same(
                cooperative.gradient[index],
                cooperative_diagonal.gradient[index],
                "CIR diagonal request changed a first derivative"
            );
            require(
                std::isfinite(scalar_diagonal.diagonal_hessian[index])
                    && std::isfinite(
                        cooperative_diagonal.diagonal_hessian[index]
                    ),
                "CIR produced a non-finite diagonal Hessian."
            );
            same(
                cooperative_diagonal.diagonal_hessian[index],
                second_only.diagonal_hessian[index],
                "CIR derivative request changed a diagonal Hessian"
            );
        }
    }
    require(
        cooperative_diagonal.stencils[
            2U * full.sensitivities.size() + 3U
        ].node_count == 4U,
        "CIR boundary diagonal did not retain four nodes."
    );

    DeviceArray<cir::ModelParameters> device_models(models);
    DeviceArray<product::RegularEuropeanSwaptionParameters>
        device_products(products);
    DeviceArray<float> legacy(models.size());
    cir::launch_cir_european_swaption_cuda<Side>(
        device_models.data,
        models.size(),
        device_products.data,
        products.size(),
        PriceConstruction::Aligned,
        models.size(),
        0U,
        models.size(),
        1.0f / 252.0f,
        128U,
        2U,
        legacy.data,
        12U,
        closed_form::WorkDistribution::cooperative
    );
    const auto legacy_prices = legacy.read();
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(
            cooperative.price[row],
            legacy_prices[row],
            "Adding CIR sensitivities changed the cooperative price bits"
        );
    }
}

}  // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        check_side<SwaptionSide::payer>();
        check_side<SwaptionSide::receiver>();
        std::cout
            << "CIR Jamshidian compact first/diagonal sensitivities passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
