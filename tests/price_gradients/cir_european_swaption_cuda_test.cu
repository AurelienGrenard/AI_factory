// CIR Jamshidian sensitivities: compact preparation, scalar/cooperative parity.
#include "common/fixed_income/price_gradients/terminal_maturity.cuh"
#include "product/european_swaption/schedule.cuh"
#include "model/fixed_income/cir/product/european_swaption.cuh"
#include "model/fixed_income/cir/product/european_swaption_price_gradients.cuh"
#include "model/fixed_income/cir/product/rate_option_price_gradients.cuh"
#include "model/fixed_income/cir/product/zero_coupon_bond_option_price_gradients.cuh"
#include "tests/price_gradients/diagonal_cuda_test_support.cuh"
#include "tests/price_gradients/closed_form_mixed_cuda_test_support.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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

__global__ void inspect_terminal_stub(
    product::RegularEuropeanSwaptionParameters parameters,
    float day_fraction,
    float terminal_payment_time,
    float* values
) {
    const auto central = product::make_european_swaption_schedule_view(
        parameters,
        product::RegularEuropeanSwaptionScheduleSource{},
        day_fraction
    );
    const auto count = central.payment_count();
    const float prefix = central.payment_time(count - 2U);
    const fixed_income::price_gradients::TerminalPaymentScheduleView<
        decltype(central)
    > schedule{central, terminal_payment_time, prefix, true};
    values[0U] = schedule.payment_time(count - 1U);
    values[1U] = schedule.accrual_fraction(count - 1U);
    values[2U] = schedule.accrual_fraction(count - 2U);
}

void check_terminal_stub_schedule() {
    constexpr float day_fraction = 1.0f / 252.0f;
    const auto& parameters = products[0U];
    const float central_terminal = static_cast<float>(
        parameters.exercise_time_days
            + parameters.payment_interval_days * parameters.payment_count
    ) * day_fraction;
    const float terminal = central_terminal + 1.0f / 504.0f;
    const float prefix = static_cast<float>(
        parameters.exercise_time_days
            + parameters.payment_interval_days
                * (parameters.payment_count - 1U)
    ) * day_fraction;
    DeviceArray<float> values(3U);
    inspect_terminal_stub<<<1U, 1U>>>(
        parameters, day_fraction, terminal, values.data
    );
    check_cuda(cudaDeviceSynchronize(), "Terminal stub schedule test");
    const auto actual = values.read();
    constexpr float tolerance = 4.0f * 1.1920929e-7f;
    require(
        std::abs(actual[0U] - terminal) <= tolerance
            && std::abs(actual[1U] - (terminal - prefix)) <= tolerance
            && std::abs(
                actual[2U] - parameters.accrual_fraction
            ) <= tolerance,
        "The maturity node did not form a terminal coupon stub."
    );
}

const pg::PriceGradientConfiguration full{{
    {"model.mean_reversion", {0.005}},
    {"model.long_term_mean", {0.005}},
    {"model.volatility", {0.005}},
    {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
    {"product.notional", {0.005}},
    {"product.strike", {0.0005, pg::BumpScale::absolute}},
    {"product.accrual_fraction", {0.005}},
    {"product.maturity_years",
     {1.0 / 504.0, pg::BumpScale::absolute}},
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

template<
    typename Plan,
    typename DiagonalLauncher,
    typename WorkspaceSizer,
    typename MixedLauncher
>
void check_scalar_terminal_maturity(
    const Plan& plan,
    DiagonalLauncher diagonal_launcher,
    WorkspaceSizer workspace_size,
    MixedLauncher mixed_launcher,
    const char* label
) {
    pg::LaunchConfiguration launch{
        pg::PricingMethod::closed_form,
        0U,
        plan.result_count,
        0U,
        128U,
        plan.sensitivity_count(),
        0U,
    };
    const auto diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(plan, launch, diagonal_launcher);
    const auto mixed = execute_closed_form_mixed(
        plan, launch, workspace_size, mixed_launcher, label
    );
    require_closed_form_diagonal_parity(diagonal, mixed, label);
    if (!(diagonal.gradient.size() == 2U
          && diagonal.diagonal_hessian.size() == 2U
          && mixed.mixed_hessians.size() == 1U
          && std::isfinite(diagonal.gradient[1U])
          && std::isfinite(diagonal.diagonal_hessian[1U])
          && std::abs(diagonal.gradient[1U]) > 1.0e-7f)) {
        throw std::runtime_error(
            std::string(label) + " did not evaluate terminal maturity."
        );
    }
}

void check_scalar_terminal_products() {
    const std::vector<cir::ModelParameters> scalar_models{models[0U]};
    const pg::PriceGradientConfiguration selection{{
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
        {"product.maturity_years",
         {1.0 / 504.0, pg::BumpScale::absolute}},
    }};

    const std::vector<product::RateOptionParameters> rate_options{{
        1.0f, 0.035f, 126U, 252U, 126U
    }};
    const auto rate_plan = cir::prepare_cir_rate_option_sensitivities(
        scalar_models,
        rate_options,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        pg::SensitivityRequest::full_hessian()
    );
    check_scalar_terminal_maturity(
        rate_plan,
        [](const auto& plan, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            cir::launch_cir_rate_option_diagonal_sensitivities_cuda<
                OptionSide::call,
                pg::SensitivityOrders::first_and_second
            >(plan, inputs, stencils, launch, outputs);
        },
        [](const auto& plan, const auto& launch) {
            return cir::cir_rate_option_mixed_node_graph_workspace_bytes<
                OptionSide::call
            >(plan, launch);
        },
        [](const auto& plan, auto inputs, auto stencils,
           auto mixed_stencils, const auto& launch, auto outputs,
           auto mixed_outputs, void* workspace, std::size_t bytes) {
            cir::launch_cir_rate_option_mixed_node_graph_sensitivities_cuda<
                OptionSide::call
            >(plan, inputs, stencils, mixed_stencils, launch, outputs,
              mixed_outputs, workspace, bytes);
        },
        "CIR rate option terminal maturity"
    );

    const std::vector<product::ZeroCouponBondOptionParameters> bond_options{{
        1.0f, 0.90f, 126U, 504U
    }};
    const auto bond_plan =
        cir::prepare_cir_zero_coupon_bond_option_sensitivities(
            scalar_models,
            bond_options,
            PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            selection,
            pg::SensitivityRequest::full_hessian()
        );
    check_scalar_terminal_maturity(
        bond_plan,
        [](const auto& plan, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            cir::
                launch_cir_zero_coupon_bond_option_diagonal_sensitivities_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(plan, inputs, stencils, launch, outputs);
        },
        [](const auto& plan, const auto& launch) {
            return cir::
                cir_zero_coupon_bond_option_mixed_node_graph_workspace_bytes<
                    OptionSide::put
                >(plan, launch);
        },
        [](const auto& plan, auto inputs, auto stencils,
           auto mixed_stencils, const auto& launch, auto outputs,
           auto mixed_outputs, void* workspace, std::size_t bytes) {
            cir::
                launch_cir_zero_coupon_bond_option_mixed_node_graph_sensitivities_cuda<
                    OptionSide::put
                >(plan, inputs, stencils, mixed_stencils, launch, outputs,
                  mixed_outputs, workspace, bytes);
        },
        "CIR zero-coupon-bond option terminal maturity"
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

    // FP32 nodal prices do not support reliable second differences here.
    const auto rejects_second_order = [](auto&& action, const char* label) {
        bool rejected = false;
        try {
            action();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, label);
    };
    rejects_second_order([&] {
        (void)prepare_diagonal<pg::SensitivityOrders::first_and_second>(full);
    }, "CIR Jamshidian accepted a diagonal Hessian request.");
    rejects_second_order([&] {
        (void)prepare_diagonal<pg::SensitivityOrders::second>(full);
    }, "CIR Jamshidian accepted a second-only request.");
    rejects_second_order([&] {
        (void)cir::prepare_cir_european_swaption_sensitivities(
            models, products, PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U}, full,
            pg::SensitivityRequest::full_hessian()
        );
    }, "CIR Jamshidian accepted a mixed Hessian request.");
    rejects_second_order([&] {
        cir::prepare_european_swaption_diagonal_sensitivity_stencils_cuda(
            first_plan, {}, {}, 0U, 0U
        );
    }, "CIR Jamshidian direct diagonal stencil preparation did not reject.");
    rejects_second_order([&] {
        cir::launch_cir_european_swaption_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >(first_plan, {}, {}, launch, {},
          closed_form::WorkDistribution::scalar);
    }, "CIR Jamshidian direct diagonal launcher did not reject.");
    rejects_second_order([&] {
        (void)cir::cir_european_swaption_mixed_node_graph_workspace_bytes<
            Side
        >(first_plan, launch);
    }, "CIR Jamshidian mixed workspace did not reject.");
    rejects_second_order([&] {
        cir::launch_cir_european_swaption_mixed_node_graph_sensitivities_cuda<
            Side
        >(first_plan, {}, {}, {}, launch, {}, {}, nullptr, 0U,
          closed_form::WorkDistribution::scalar);
    }, "CIR Jamshidian direct mixed launcher did not reject.");

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
        check_terminal_stub_schedule();
        check_scalar_terminal_products();
        check_side<SwaptionSide::payer>();
        check_side<SwaptionSide::receiver>();
        std::cout
            << "CIR Jamshidian first-order sensitivities and Hessian rejection passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
