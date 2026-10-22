// Bates American sensitivities: frozen-policy replay and jump-event coupling.
#include "model/equity/markovian/bates/product/american_option.cuh"
#include "model/equity/markovian/bates/product/american_option_price_delta.cuh"
#include "model/equity/markovian/bates/product/american_option_price_gradients.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_frozen_exercise_replay.cuh"
#include "model/equity/markovian/bates/dynamics_impl.cuh"
#include "model/equity/markovian/bates/price_gradients/coupled_dynamics_impl.cuh"
#include "tests/price_gradients/cuda_test_support.cuh"
#include "tests/price_gradients/mixed_node_graph_cuda_test_support.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace bates = model::equity::bates;
namespace pg = price_gradients;
using price_gradient_test::DeviceArray;
using price_gradient_test::require;
using price_gradient_test::same;

std::size_t test_paths = 4096U;

struct Run {
    std::vector<float> prices;
    std::vector<float> price_errors;
    std::vector<float> gradients;
    std::vector<float> gradient_errors;
    std::vector<float> hessians;
    std::vector<float> hessian_errors;
    longstaff_schwartz::LaunchResult launch;
};

using Replay = longstaff_schwartz::price_gradients::
    DevicePreparedFrozenExerciseReplay<
        bates::price_gradients::CoupledDynamics,
        3U
    >;

__global__ void canonical_replay_kernel(
    bates::ModelParameters model,
    pg::TimeConfiguration time,
    std::uint32_t initial_steps,
    std::uint32_t regular_steps,
    std::uint32_t observation,
    std::uint64_t seed,
    float* values
) {
    using Plan = bates::AmericanOptionPriceGradientPlan;
    using Scenario = Plan::Preparation::Scenario;
    pg::SensitivityNodes<Scenario, 3U> nodes{};
    Scenario central{};
    Plan::Preparation::make_central(
        model,
        {1.0f, 63U, 7U},
        time,
        central
    );
    nodes[0U] = central;
    nodes[1U] = central;
    nodes[2U] = central;
    nodes[1U].reuse_central = false;
    nodes[2U].reuse_central = false;
    const auto prepared = Replay::prepare(
        nodes,
        3U,
        {time.dt, 6.0f * time.dt, 14.0f * time.dt}
    );

    const auto canonical_dynamics = bates::prepare_model(model, time.dt);
    auto canonical_state = bates::initial_state(canonical_dynamics);
    const auto key = philox::make_key(seed);
    bates::DynamicsPolicy::RandomContext random(key, 0U);
    bates::DynamicsPolicy::advance(
        canonical_dynamics, initial_steps, random, canonical_state
    );
    for (std::uint32_t index = 0U; index < observation; ++index) {
        bates::DynamicsPolicy::advance(
            canonical_dynamics, regular_steps, random, canonical_state
        );
    }
    const float canonical_spot = bates::DynamicsPolicy::spot(canonical_state);
    const auto replayed = Replay::evaluate(
        prepared,
        {observation, canonical_spot},
        key,
        0U,
        initial_steps,
        regular_steps
    );
    values[0U] = canonical_spot;
    values[1U] = replayed[0U];
    values[2U] = replayed[1U];
    values[3U] = replayed[2U];
}

void verify_canonical_replay() {
    const bates::ModelParameters model{
        1.00f, .03f, .01f, .04f, 1.5f, .04f, .40f, -.7f,
        .75f, -.08f, .18f,
    };
    DeviceArray<float> device_values(4U);
    canonical_replay_kernel<<<1U, 1U>>>(
        model,
        {1.0f / 504.0f, 2U},
        6U,
        14U,
        3U,
        77191U,
        device_values.data
    );
    check_cuda(cudaGetLastError(), "launch canonical Bates replay oracle");
    const auto values = device_values.read();
    same(values[1U], values[0U], "Frozen trace changed canonical Bates spot");
    same(values[2U], values[0U], "First Bates replay node changed RNG mapping");
    same(values[3U], values[0U], "Second Bates replay node changed RNG mapping");
}

void close_relative(
    float actual,
    float expected,
    float tolerance,
    const char* message
) {
    const float scale = std::max(std::abs(expected), 1.0e-12f);
    if (std::abs(actual - expected) > tolerance * scale) {
        std::cerr << message << ": " << std::hexfloat << actual
                  << " vs " << expected << std::defaultfloat << '\n';
        throw std::runtime_error(message);
    }
}

template<pg::SensitivityOrders Orders, OptionSide Side>
Run execute(
    const bates::AmericanOptionPriceGradientPlan& plan,
    std::size_t paths,
    unsigned int threads,
    std::size_t blocks,
    std::uint64_t seed,
    bool node_graph = false,
    longstaff_schwartz::price_gradients::ExerciseReplayStrategy replay =
        longstaff_schwartz::price_gradients::ExerciseReplayStrategy::
            frozen_exercise_time
) {
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    const std::size_t rows = plan.result_count;
    const std::size_t sensitivities = plan.sensitivity_count();
    DeviceArray<bates::AmericanOptionPriceGradientPlan::Model>
        models(plan.models);
    DeviceArray<bates::AmericanOptionPriceGradientPlan::Product>
        products(plan.products);
    DeviceArray<bates::AmericanOptionPriceGradientPlan::SensitivitySpec>
        sensitivity_specs(plan.sensitivities);
    DeviceArray<pg::SensitivityStencil<node_capacity>> stencils(
        rows * sensitivities
    );
    DeviceArray<equity::price_gradients::device_preparation::Error> error(1U);
    DeviceArray<float> prices(rows);
    DeviceArray<float> price_errors(rows);
    DeviceArray<float> gradients(
        pg::requests_first_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> gradient_errors(
        pg::requests_first_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> hessians(
        pg::requests_second_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> hessian_errors(
        pg::requests_second_v<Orders> ? rows * sensitivities : 0U
    );
    const bates::AmericanOptionPriceGradientPlan::DeviceInputs inputs{
        models.data,
        models.count,
        products.data,
        products.count,
        sensitivity_specs.data,
        sensitivity_specs.count,
    };
    const monte_carlo::price_gradients::DevicePreparedStencilOutputs<
        node_capacity
    > stencil_outputs{
        stencils.data,
        stencils.count,
        error.data,
    };
    const pg::SensitivityOutputs outputs{
        prices.data,
        price_errors.data,
        gradients.data,
        gradient_errors.data,
        hessians.data,
        hessian_errors.data,
        rows,
        rows * sensitivities,
    };
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        paths,
        threads,
        blocks,
        seed,
        1U,
    };
    std::size_t workspace_bytes = 0U;
    if constexpr (pg::requests_second_v<Orders>) {
        if (node_graph) {
            workspace_bytes =
                bates::bates_american_option_node_graph_workspace_bytes_with_replay<
                    Side, Orders
                >(plan, launch, replay);
        }
    }
    DeviceArray<std::uint8_t> workspace(workspace_bytes);
    const auto result = [&] {
        if constexpr (Orders == pg::SensitivityOrders::first) {
            return bates::launch_bates_american_option_price_gradients_with_replay_cuda<
                Side
            >(
                plan,
                inputs,
                stencil_outputs,
                launch,
                {
                    outputs.prices,
                    outputs.price_standard_errors,
                    outputs.gradients,
                    outputs.gradient_standard_errors,
                    outputs.price_capacity,
                    outputs.sensitivity_capacity,
                },
                replay
            );
        } else {
            if (node_graph) {
                return bates::
                    launch_bates_american_option_node_graph_sensitivities_with_replay_cuda<
                        Side,
                        Orders
                    >(
                        plan,
                        inputs,
                        stencil_outputs,
                        launch,
                        outputs,
                        workspace.data,
                        workspace.count,
                        replay
                    );
            }
            return bates::
                launch_bates_american_option_diagonal_sensitivities_with_replay_cuda<
                    Side,
                    Orders
                >(
                    plan,
                    inputs,
                    stencil_outputs,
                    launch,
                    outputs,
                    replay
                );
        }
    }();
    longstaff_schwartz::validate_regression_diagnostics(
        result, "Bates American sensitivities test"
    );
    require(
        error.read()[0U].code == 0,
        "Bates American device sensitivity preparation failed."
    );
    return {
        prices.read(),
        price_errors.read(),
        gradients.read(),
        gradient_errors.read(),
        hessians.read(),
        hessian_errors.read(),
        result,
    };
}

template<OptionSide Side>
void run() {
    constexpr std::size_t rows = 2U;
    constexpr std::size_t blocks = 4U;
    constexpr unsigned int threads = 128U;
    constexpr std::uint64_t seed = 53981U;
    const std::vector<bates::ModelParameters> models{
        {
            1.00f, .03f, .01f, .04f, 1.5f, .04f, .40f, -.7f,
            .45f, -.08f, .18f,
        },
        {
            .85f, .01f, .00f, .07f, 1.1f, .05f, .35f, -.4f,
            .80f, -.12f, .25f,
        },
    };
    const std::vector<product::AmericanOptionParameters> products{
        {1.00f, 63U, 7U},
        {.95f, 35U, 7U},
    };
    const pg::TimeConfiguration time{1.0f / 504.0f, 2U};
    const pg::Sensitivity spot{"model.spot", {.005f}};
    const pg::PriceGradientConfiguration full{{
        spot,
        {"model.risk_free_rate", {.0005f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {.0005f, pg::BumpScale::absolute}},
        {"model.initial_variance", {.001f, pg::BumpScale::absolute}},
        {"model.kappa", {.005f}},
        {"model.theta", {.005f}},
        {"model.gamma", {.005f}},
        {"model.rho", {.002f, pg::BumpScale::absolute}},
        {"model.jump_intensity", {.05f, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {.002f, pg::BumpScale::absolute}},
        {"model.jump_log_volatility", {.005f}},
        {"product.strike", {.005f}},
    }};
    const auto prepare_first = [&](const pg::PriceGradientConfiguration& c) {
        return bates::prepare_bates_american_option_price_gradients(
            models,
            products,
            PriceConstruction::Aligned,
            time,
            c
        );
    };

    const auto price_only = execute<pg::SensitivityOrders::first, Side>(
        prepare_first({}), test_paths, threads, blocks, seed
    );
    const auto spot_only = execute<pg::SensitivityOrders::first, Side>(
        prepare_first({{spot}}), test_paths, threads, blocks, seed
    );
    const auto selected = execute<pg::SensitivityOrders::first, Side>(
        prepare_first(full), test_paths, threads, blocks, seed
    );

    DeviceArray<bates::ModelParameters> device_models(models);
    DeviceArray<product::AmericanOptionParameters> device_products(products);
    DeviceArray<float> legacy(4U * rows);
    const auto delta_launch =
        bates::launch_bates_american_option_price_delta_cuda<Side>(
            models.data(),
            device_models.data,
            rows,
            products.data(),
            device_products.data,
            rows,
            PriceConstruction::Aligned,
            rows,
            test_paths,
            time.dt,
            time.simulation_steps_per_day,
            threads,
            blocks,
            seed,
            equity::price_delta::SpotBumpConfiguration{.01f},
            legacy.data,
            legacy.data + rows,
            legacy.data + 2U * rows,
            legacy.data + 3U * rows
        );
    longstaff_schwartz::validate_regression_diagnostics(
        delta_launch, "Bates American price-delta reference"
    );
    const auto reference = legacy.read();
    for (std::size_t row = 0U; row < rows; ++row) {
        same(
            price_only.prices[row],
            spot_only.prices[row],
            "Bates American empty selection changed central price"
        );
        same(
            price_only.price_errors[row],
            spot_only.price_errors[row],
            "Bates American empty selection changed central error"
        );
        same(
            spot_only.prices[row],
            reference[row],
            "Bates American central price differs from price-delta"
        );
        same(
            spot_only.price_errors[row],
            reference[rows + row],
            "Bates American central error differs from price-delta"
        );
        close_relative(
            spot_only.gradients[row],
            reference[2U * rows + row],
            5.0e-6f,
            "Bates American spot sensitivity differs from price-delta"
        );
        close_relative(
            spot_only.gradient_errors[row],
            reference[3U * rows + row],
            5.0e-5f,
            "Bates American spot error differs from price-delta"
        );
        same(
            selected.prices[row],
            spot_only.prices[row],
            "Adding Bates American sensitivities changed central price"
        );
        same(
            selected.gradients[row * full.sensitivities.size()],
            spot_only.gradients[row],
            "Adding Bates American sensitivities changed spot sensitivity"
        );
    }
    for (float value : selected.gradients) {
        require(std::isfinite(value), "Non-finite Bates American gradient.");
    }
    for (float value : selected.gradient_errors) {
        require(
            std::isfinite(value) && value >= 0.0f,
            "Invalid Bates American gradient error."
        );
    }

    constexpr std::size_t jump_intensity = 8U;
    const auto jump_only = execute<pg::SensitivityOrders::first, Side>(
        prepare_first({{full.sensitivities[jump_intensity]}}),
        test_paths,
        threads,
        blocks,
        seed
    );
    for (std::size_t row = 0U; row < rows; ++row) {
        same(
            jump_only.gradients[row],
            selected.gradients[
                row * full.sensitivities.size() + jump_intensity
            ],
            "Bates American selection changed jump-intensity sensitivity"
        );
        same(
            jump_only.gradient_errors[row],
            selected.gradient_errors[
                row * full.sensitivities.size() + jump_intensity
            ],
            "Bates American selection changed jump-intensity error"
        );
    }

    const auto diagonal_plan =
        bates::prepare_bates_american_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            time,
            full,
            {pg::SensitivityOrders::first_and_second}
        );
    const auto diagonal = execute<
        pg::SensitivityOrders::first_and_second,
        Side
    >(diagonal_plan, test_paths, threads, blocks, seed);
    const auto node_graph = execute<
        pg::SensitivityOrders::first_and_second,
        Side
    >(diagonal_plan, test_paths, threads, blocks, seed, true);
    require(
        node_graph.prices == diagonal.prices
            && node_graph.price_errors == diagonal.price_errors
            && node_graph.gradients == diagonal.gradients
            && node_graph.gradient_errors == diagonal.gradient_errors
            && node_graph.hessians == diagonal.hessians
            && node_graph.hessian_errors == diagonal.hessian_errors,
        "Bates American mono and node_graph sensitivity outputs differ."
    );
    constexpr auto frozen_policy_replay =
        longstaff_schwartz::price_gradients::ExerciseReplayStrategy::
            frozen_regression_policy;
    const auto frozen_policy = execute<
        pg::SensitivityOrders::first_and_second,
        Side
    >(
        diagonal_plan,
        test_paths,
        threads,
        blocks,
        seed,
        false,
        frozen_policy_replay
    );
    const auto frozen_policy_graph = execute<
        pg::SensitivityOrders::first_and_second,
        Side
    >(
        diagonal_plan,
        test_paths,
        threads,
        blocks,
        seed,
        true,
        frozen_policy_replay
    );
    require(
        frozen_policy_graph.prices == frozen_policy.prices
            && frozen_policy_graph.price_errors == frozen_policy.price_errors
            && frozen_policy_graph.gradients == frozen_policy.gradients
            && frozen_policy_graph.gradient_errors
                == frozen_policy.gradient_errors
            && frozen_policy_graph.hessians == frozen_policy.hessians
            && frozen_policy_graph.hessian_errors
                == frozen_policy.hessian_errors,
        "Bates frozen-policy mono and node_graph outputs differ."
    );
    require(
        frozen_policy.prices == diagonal.prices
            && frozen_policy.price_errors == diagonal.price_errors,
        "Bates replay strategy changed central results."
    );
    require(
        diagonal.prices == selected.prices
            && diagonal.price_errors == selected.price_errors
            && diagonal.gradients == selected.gradients
            && diagonal.gradient_errors == selected.gradient_errors,
        "Bates American diagonal request changed price or gradient outputs."
    );
    for (float value : diagonal.hessians) {
        require(
            std::isfinite(value),
            "Non-finite Bates American diagonal Hessian."
        );
    }
    for (float value : diagonal.hessian_errors) {
        require(
            std::isfinite(value) && value >= 0.0f,
            "Invalid Bates American diagonal Hessian error."
        );
    }

    const auto mixed_plan =
        bates::prepare_bates_american_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            time,
            full,
            pg::SensitivityRequest::full_hessian()
        );
    price_gradient_test::require_mixed_node_graph_parity(
        mixed_plan,
        [](const auto& p, auto inputs, auto stencils,
           const auto& configuration, auto outputs) {
            return bates::
                launch_bates_american_option_diagonal_sensitivities_cuda<
                    Side, pg::SensitivityOrders::first_and_second
                >(p, inputs, stencils, configuration, outputs);
        },
        [](const auto& p, const auto& configuration) {
            return bates::
                bates_american_option_mixed_node_graph_workspace_bytes<Side>(
                    p, configuration
                );
        },
        [](const auto& p, auto inputs, auto stencils,
           auto mixed_stencils, const auto& configuration,
           auto outputs, auto mixed_outputs,
           void* workspace, std::size_t workspace_bytes) {
            return bates::
                launch_bates_american_option_mixed_node_graph_sensitivities_cuda<
                    Side
                >(
                    p, inputs, stencils, mixed_stencils,
                    configuration, outputs, mixed_outputs,
                    workspace, workspace_bytes
                );
        },
        257U,
        seed,
        "Bates American mixed node graph"
    );
    price_gradient_test::require_mixed_node_graph_parity(
        mixed_plan,
        [](const auto& p, auto inputs, auto stencils,
           const auto& configuration, auto outputs) {
            return bates::
                launch_bates_american_option_diagonal_sensitivities_with_replay_cuda<
                    Side, pg::SensitivityOrders::first_and_second
                >(
                    p, inputs, stencils, configuration, outputs,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        },
        [](const auto& p, const auto& configuration) {
            return bates::
                bates_american_option_mixed_node_graph_workspace_bytes_with_replay<
                    Side
                >(
                    p,
                    configuration,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        },
        [](const auto& p, auto inputs, auto stencils,
           auto mixed_stencils, const auto& configuration,
           auto outputs, auto mixed_outputs,
           void* workspace, std::size_t workspace_bytes) {
            return bates::
                launch_bates_american_option_mixed_node_graph_sensitivities_with_replay_cuda<
                    Side
                >(
                    p, inputs, stencils, mixed_stencils,
                    configuration, outputs, mixed_outputs,
                    workspace, workspace_bytes,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        },
        257U,
        seed,
        "Bates American frozen-policy mixed node graph"
    );
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            test_paths = 257U;
        } else if (argc != 1) {
            throw std::invalid_argument(
                "Usage: test_price_gradients_bates_american_cuda "
                "[--sanitizer]"
            );
        }
        verify_canonical_replay();
        run<OptionSide::call>();
        run<OptionSide::put>();
        std::cout << "Bates American frozen sensitivities passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
