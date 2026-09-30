// Frozen-exercise versus frozen-policy diagonal-sensitivity benchmark.
#include "tests/performance/benchmark_support.cuh"

#include "model/equity/markovian/bates/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"
#include "model/fixed_income/cir/product/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/g2/product/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/product/svensson/bermudan_swaption_price_gradients.cuh"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace perf = performance;
namespace pg = price_gradients;
namespace lspg = longstaff_schwartz::price_gradients;

constexpr auto kOrders = pg::SensitivityOrders::first_and_second;
constexpr unsigned int kThreads = 128U;
constexpr std::size_t kBlocks = 4U;
constexpr std::uint64_t kSeed = 2'170'000'031ULL;
constexpr int kWarmups = 20;
constexpr int kRepetitions = 11;
constexpr std::size_t kOperationsPerSample = 64U;

enum class ExecutionStrategy { mono, node_graph };

const char* replay_name(lspg::ExerciseReplayStrategy replay) {
    switch (replay) {
    case lspg::ExerciseReplayStrategy::frozen_exercise_time:
        return "frozen_exercise";
    case lspg::ExerciseReplayStrategy::frozen_regression_policy:
        return "frozen_policy";
    }
    throw std::invalid_argument("Unknown exercise replay strategy.");
}

const char* execution_name(ExecutionStrategy execution) {
    switch (execution) {
    case ExecutionStrategy::mono:
        return "mono";
    case ExecutionStrategy::node_graph:
        return "node_graph";
    }
    throw std::invalid_argument("Unknown execution strategy.");
}

lspg::ExerciseReplayStrategy parse_replay(std::string_view value) {
    if (value == "frozen_exercise") {
        return lspg::ExerciseReplayStrategy::frozen_exercise_time;
    }
    if (value == "frozen_policy") {
        return lspg::ExerciseReplayStrategy::frozen_regression_policy;
    }
    throw std::invalid_argument(
        "REPLAY must be frozen_exercise or frozen_policy."
    );
}

ExecutionStrategy parse_execution(std::string_view value) {
    if (value == "mono") return ExecutionStrategy::mono;
    if (value == "node_graph") return ExecutionStrategy::node_graph;
    throw std::invalid_argument("EXECUTION must be mono or node_graph.");
}

template<typename Plan, bool HasCurve = requires { typename Plan::Curve; }>
class DeviceInputs;

template<typename Plan>
class DeviceInputs<Plan, false> {
public:
    explicit DeviceInputs(const Plan& plan)
        : models_(plan.models.size() * sizeof(typename Plan::Model),
                  perf::DeviceMemoryRole::persistent_input),
          products_(plan.products.size() * sizeof(typename Plan::Product),
                    perf::DeviceMemoryRole::persistent_input),
          sensitivities_(
              plan.sensitivities.size()
                  * sizeof(typename Plan::SensitivitySpec),
              perf::DeviceMemoryRole::persistent_input
          ) {
        perf::copy_to_device(models_, plan.models);
        perf::copy_to_device(products_, plan.products);
        perf::copy_to_device(sensitivities_, plan.sensitivities);
    }

    typename Plan::DeviceInputs get(const Plan& plan) {
        return {
            models_.template as<typename Plan::Model>(),
            plan.models.size(),
            products_.template as<typename Plan::Product>(),
            plan.products.size(),
            sensitivities_.template as<typename Plan::SensitivitySpec>(),
            plan.sensitivities.size(),
        };
    }

private:
    perf::DeviceBuffer models_;
    perf::DeviceBuffer products_;
    perf::DeviceBuffer sensitivities_;
};

template<typename Plan>
class DeviceInputs<Plan, true> {
public:
    explicit DeviceInputs(const Plan& plan)
        : models_(plan.models.size() * sizeof(typename Plan::Model),
                  perf::DeviceMemoryRole::persistent_input),
          curves_(plan.curves.size() * sizeof(typename Plan::Curve),
                  perf::DeviceMemoryRole::persistent_input),
          products_(plan.products.size() * sizeof(typename Plan::Product),
                    perf::DeviceMemoryRole::persistent_input),
          sensitivities_(
              plan.sensitivities.size()
                  * sizeof(typename Plan::SensitivitySpec),
              perf::DeviceMemoryRole::persistent_input
          ) {
        perf::copy_to_device(models_, plan.models);
        perf::copy_to_device(curves_, plan.curves);
        perf::copy_to_device(products_, plan.products);
        perf::copy_to_device(sensitivities_, plan.sensitivities);
    }

    typename Plan::DeviceInputs get(const Plan& plan) {
        return {
            models_.template as<typename Plan::Model>(),
            plan.models.size(),
            curves_.template as<typename Plan::Curve>(),
            plan.curves.size(),
            products_.template as<typename Plan::Product>(),
            plan.products.size(),
            sensitivities_.template as<typename Plan::SensitivitySpec>(),
            plan.sensitivities.size(),
        };
    }

private:
    perf::DeviceBuffer models_;
    perf::DeviceBuffer curves_;
    perf::DeviceBuffer products_;
    perf::DeviceBuffer sensitivities_;
};

std::uint64_t finite_output_hash(const std::vector<float>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Non-finite early-exercise replay benchmark output."
            );
        }
        hash ^= std::bit_cast<std::uint32_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

template<
    typename Plan,
    typename MonoLaunch,
    typename GraphWorkspaceBytes,
    typename GraphLaunch>
int benchmark_diagonal(
    std::string_view model,
    const Plan& plan,
    std::size_t paths,
    lspg::ExerciseReplayStrategy replay,
    ExecutionStrategy execution,
    MonoLaunch mono_launch,
    GraphWorkspaceBytes graph_workspace_bytes,
    GraphLaunch graph_launch
) {
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<kOrders>::node_capacity;
    const std::size_t rows = plan.result_count;
    const std::size_t sensitivity_count = plan.sensitivity_count();
    const std::size_t sensitivity_values = rows * sensitivity_count;
    const std::size_t output_count = 2U * rows + 4U * sensitivity_values;
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        paths,
        kThreads,
        kBlocks,
        kSeed,
        1U,
    };

    DeviceInputs<Plan> device_inputs(plan);
    perf::DeviceBuffer stencil_buffer(
        sensitivity_values
            * sizeof(pg::SensitivityStencil<node_capacity>),
        perf::DeviceMemoryRole::output
    );
    perf::DeviceBuffer preparation_error_buffer(
        sizeof(pg::device_preparation::Error),
        perf::DeviceMemoryRole::output
    );
    perf::DeviceBuffer output_buffer(
        output_count * sizeof(float),
        perf::DeviceMemoryRole::output
    );

    const auto inputs = device_inputs.get(plan);
    const typename Plan::DiagonalStencilOutputs stencil_outputs{
        stencil_buffer.template as<pg::SensitivityStencil<node_capacity>>(),
        sensitivity_values,
        preparation_error_buffer.template as<
            pg::device_preparation::Error
        >(),
    };
    float* output = output_buffer.template as<float>();
    float* gradient = output + 2U * rows;
    float* gradient_error = gradient + sensitivity_values;
    float* hessian = gradient_error + sensitivity_values;
    float* hessian_error = hessian + sensitivity_values;
    const pg::SensitivityOutputs outputs{
        output,
        output + rows,
        gradient,
        gradient_error,
        hessian,
        hessian_error,
        rows,
        sensitivity_values,
    };

    const std::size_t external_workspace_bytes =
        execution == ExecutionStrategy::node_graph
            ? graph_workspace_bytes(plan, launch, replay)
            : 0U;
    perf::DeviceBuffer external_workspace(
        std::max<std::size_t>(external_workspace_bytes, 1U),
        perf::DeviceMemoryRole::caller_workspace
    );

    longstaff_schwartz::LaunchResult latest{};
    const auto invoke = [&]() {
        if (execution == ExecutionStrategy::mono) {
            latest = mono_launch(
                plan,
                inputs,
                stencil_outputs,
                launch,
                outputs,
                replay
            );
        } else {
            latest = graph_launch(
                plan,
                inputs,
                stencil_outputs,
                launch,
                outputs,
                external_workspace.template as<void>(),
                external_workspace_bytes,
                replay
            );
        }
        longstaff_schwartz::validate_regression_diagnostics(
            latest, "early-exercise replay performance benchmark"
        );
        return latest.kernel_seconds * 1'000.0;
    };

    const bool profile_probe =
        std::getenv("AI_FACTORY_PERFORMANCE_PROFILE_PROBE") != nullptr;
    const int warmups = profile_probe ? 1 : kWarmups;
    const int repetitions = profile_probe ? 3 : kRepetitions;
    const std::size_t operations_per_sample =
        profile_probe ? 1U : kOperationsPerSample;
    const auto measurement = perf::measure_synchronous_cuda_pipeline(
        invoke,
        warmups,
        repetitions,
        operations_per_sample
    );
    invoke();
    const auto preparation_error =
        perf::copy_from_device<pg::device_preparation::Error>(
            preparation_error_buffer, 1U
        )[0U];
    if (preparation_error.code != pg::device_preparation::valid) {
        throw std::runtime_error(
            "Early-exercise replay benchmark stencil preparation failed."
        );
    }
    const auto host_output = perf::copy_from_device<float>(
        output_buffer, output_count
    );

    const std::string variant = std::string(model) + "/"
        + replay_name(replay) + "/" + execution_name(execution);
    perf::emit_measurement(
        "FROZEN-REPLAY-QUALIFICATION",
        "price_gradients_early_exercise_replay",
        variant,
        measurement,
        {
            {"model", model},
            {"replay", replay_name(replay)},
            {"execution", execution_name(execution)},
            {"rows", rows},
            {"sensitivity_count", sensitivity_count},
            {"paths_per_price", paths},
            {"threads_per_block", kThreads},
            {"blocks_per_price", kBlocks},
            {"diagonal_node_capacity", node_capacity},
            {"internal_lsm_workspace_bytes", latest.workspace_bytes},
            {"external_node_graph_workspace_bytes",
                external_workspace_bytes},
            {"kernel_launch_count", latest.kernel_launch_count},
        },
        {
            {"finite", true},
            {"output_hash", finite_output_hash(host_output)},
        },
        warmups,
        repetitions,
        {},
        latest.workspace_bytes
    );
    return 0;
}

int benchmark_black_scholes(
    std::size_t paths,
    lspg::ExerciseReplayStrategy replay,
    ExecutionStrategy execution
) {
    namespace model = model::equity::black_scholes;
    const std::vector<model::ModelParameters> models{{
        1.0f, 0.03f, 0.01f, 0.20f
    }};
    const std::vector<product::AmericanOptionParameters> products{{
        1.0f, 63U, 7U
    }};
    const pg::PriceGradientConfiguration sensitivities{{
        {"model.spot", {0.005f}},
        {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
        {"model.volatility", {0.005f}},
        {"product.strike", {0.005f}},
    }};
    const auto plan =
        model::prepare_black_scholes_american_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            sensitivities,
            {kOrders}
        );
    return benchmark_diagonal(
        "black_scholes", plan, paths, replay, execution,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, auto selected_replay) {
            return model::
                launch_black_scholes_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::put, kOrders
                >(
                    host, inputs, stencils, launch, outputs, selected_replay
                );
        },
        [](const auto& host, const auto& launch, auto selected_replay) {
            return model::
                black_scholes_american_option_node_graph_workspace_bytes_with_replay<
                    OptionSide::put, kOrders
                >(host, launch, selected_replay);
        },
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, void* workspace,
           std::size_t workspace_bytes, auto selected_replay) {
            return model::
                launch_black_scholes_american_option_node_graph_sensitivities_with_replay_cuda<
                    OptionSide::put, kOrders
                >(
                    host, inputs, stencils, launch, outputs,
                    workspace, workspace_bytes, selected_replay
                );
        }
    );
}

int benchmark_heston(
    std::size_t paths,
    lspg::ExerciseReplayStrategy replay,
    ExecutionStrategy execution
) {
    namespace model = model::equity::heston;
    const std::vector<model::ModelParameters> models{{
        1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f
    }};
    const std::vector<product::AmericanOptionParameters> products{{
        1.0f, 63U, 7U
    }};
    const pg::PriceGradientConfiguration sensitivities{{
        {"model.spot", {0.005f}},
        {"model.initial_variance", {0.001f, pg::BumpScale::absolute}},
        {"model.rho", {0.002f, pg::BumpScale::absolute}},
        {"product.strike", {0.005f}},
    }};
    const auto plan = model::prepare_heston_american_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        sensitivities,
        {kOrders}
    );
    return benchmark_diagonal(
        "heston", plan, paths, replay, execution,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, auto selected_replay) {
            return model::
                launch_heston_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::put, kOrders
                >(
                    host, inputs, stencils, launch, outputs, selected_replay
                );
        },
        [](const auto& host, const auto& launch, auto selected_replay) {
            return model::
                heston_american_option_node_graph_workspace_bytes_with_replay<
                    OptionSide::put, kOrders
                >(host, launch, selected_replay);
        },
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, void* workspace,
           std::size_t workspace_bytes, auto selected_replay) {
            return model::
                launch_heston_american_option_node_graph_sensitivities_with_replay_cuda<
                    OptionSide::put, kOrders
                >(
                    host, inputs, stencils, launch, outputs,
                    workspace, workspace_bytes, selected_replay
                );
        }
    );
}

int benchmark_bates(
    std::size_t paths,
    lspg::ExerciseReplayStrategy replay,
    ExecutionStrategy execution
) {
    namespace model = model::equity::bates;
    const std::vector<model::ModelParameters> models{{
        1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.40f, -0.70f,
        0.45f, -0.08f, 0.18f
    }};
    const std::vector<product::AmericanOptionParameters> products{{
        1.0f, 63U, 7U
    }};
    const pg::PriceGradientConfiguration sensitivities{{
        {"model.spot", {0.005f}},
        {"model.initial_variance", {0.001f, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05f, pg::BumpScale::absolute}},
        {"product.strike", {0.005f}},
    }};
    const auto plan = model::prepare_bates_american_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        sensitivities,
        {kOrders}
    );
    return benchmark_diagonal(
        "bates", plan, paths, replay, execution,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, auto selected_replay) {
            return model::
                launch_bates_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::put, kOrders
                >(
                    host, inputs, stencils, launch, outputs, selected_replay
                );
        },
        [](const auto& host, const auto& launch, auto selected_replay) {
            return model::
                bates_american_option_node_graph_workspace_bytes_with_replay<
                    OptionSide::put, kOrders
                >(host, launch, selected_replay);
        },
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, void* workspace,
           std::size_t workspace_bytes, auto selected_replay) {
            return model::
                launch_bates_american_option_node_graph_sensitivities_with_replay_cuda<
                    OptionSide::put, kOrders
                >(
                    host, inputs, stencils, launch, outputs,
                    workspace, workspace_bytes, selected_replay
                );
        }
    );
}

const std::vector<product::BermudanSwaptionParameters> kSwaptions{{
    1.0f, 0.03f, 0.5f, 126U, 126U, 6U, 4U
}};

int benchmark_cir(
    std::size_t paths,
    lspg::ExerciseReplayStrategy replay,
    ExecutionStrategy execution
) {
    namespace model = model::fixed_income::cir;
    const std::vector<model::ModelParameters> models{{
        {0.50f, 0.040f, 0.10f}, 0.040f
    }};
    const pg::PriceGradientConfiguration sensitivities{{
        {"model.mean_reversion", {0.005f}},
        {"model.initial_state", {0.0005f, pg::BumpScale::absolute}},
        {"product.strike", {0.0005f, pg::BumpScale::absolute}},
    }};
    const auto plan = model::prepare_cir_bermudan_swaption_sensitivities(
        models,
        kSwaptions,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        sensitivities,
        {kOrders}
    );
    return benchmark_diagonal(
        "cir", plan, paths, replay, execution,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, auto selected_replay) {
            return model::
                launch_cir_bermudan_swaption_diagonal_sensitivities_with_replay_cuda<
                    SwaptionSide::payer, kOrders
                >(
                    host, inputs, stencils, launch, outputs, selected_replay
                );
        },
        [](const auto& host, const auto& launch, auto selected_replay) {
            return model::
                cir_bermudan_swaption_node_graph_workspace_bytes_with_replay<
                    SwaptionSide::payer, kOrders
                >(host, launch, selected_replay);
        },
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, void* workspace,
           std::size_t workspace_bytes, auto selected_replay) {
            return model::
                launch_cir_bermudan_swaption_node_graph_sensitivities_with_replay_cuda<
                    SwaptionSide::payer, kOrders
                >(
                    host, inputs, stencils, launch, outputs,
                    workspace, workspace_bytes, selected_replay
                );
        }
    );
}

int benchmark_g2(
    std::size_t paths,
    lspg::ExerciseReplayStrategy replay,
    ExecutionStrategy execution
) {
    namespace model = model::fixed_income::g2;
    const std::vector<model::ModelParameters> models{{
        {0.19f, 0.010f, 0.47f, 0.014f, 0.20f},
        {0.030f, 0.010f}
    }};
    const pg::PriceGradientConfiguration sensitivities{{
        {"model.mean_reversion_x", {0.005f}},
        {"model.volatility_y", {0.0005f, pg::BumpScale::absolute}},
        {"model.correlation", {0.005f, pg::BumpScale::absolute}},
        {"product.strike", {0.0005f, pg::BumpScale::absolute}},
    }};
    const auto plan = model::prepare_g2_bermudan_swaption_sensitivities(
        models,
        kSwaptions,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        sensitivities,
        {kOrders}
    );
    return benchmark_diagonal(
        "g2", plan, paths, replay, execution,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, auto selected_replay) {
            return model::
                launch_g2_bermudan_swaption_diagonal_sensitivities_with_replay_cuda<
                    SwaptionSide::payer, kOrders
                >(
                    host, inputs, stencils, launch, outputs, selected_replay
                );
        },
        [](const auto& host, const auto& launch, auto selected_replay) {
            return model::
                g2_bermudan_swaption_node_graph_workspace_bytes_with_replay<
                    SwaptionSide::payer, kOrders
                >(host, launch, selected_replay);
        },
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, void* workspace,
           std::size_t workspace_bytes, auto selected_replay) {
            return model::
                launch_g2_bermudan_swaption_node_graph_sensitivities_with_replay_cuda<
                    SwaptionSide::payer, kOrders
                >(
                    host, inputs, stencils, launch, outputs,
                    workspace, workspace_bytes, selected_replay
                );
        }
    );
}

int benchmark_g2_plus_plus_svensson(
    std::size_t paths,
    lspg::ExerciseReplayStrategy replay,
    ExecutionStrategy execution
) {
    namespace model = model::fixed_income::g2_plus_plus::svensson;
    const std::vector<
        ai_factory::workbench::model::fixed_income::g2_plus_plus::ModelParameters
    > models{{
        {0.20f, 0.005f, 0.75f, 0.012f, -0.40f}
    }};
    const std::vector<curve::svensson::SvenssonParameters> curves{{
        0.030f, -0.010f, 0.010f, 0.005f, 2.0f, 5.0f
    }};
    const pg::PriceGradientConfiguration sensitivities{{
        {"model.mean_reversion_x", {0.005f}},
        {"model.volatility_x", {0.0005f, pg::BumpScale::absolute}},
        {"model.mean_reversion_y", {0.005f}},
        {"model.volatility_y", {0.0005f, pg::BumpScale::absolute}},
        {"model.correlation", {0.005f, pg::BumpScale::absolute}},
        {"curve.beta0", {0.0005f, pg::BumpScale::absolute}},
        {"curve.beta1", {0.0005f, pg::BumpScale::absolute}},
        {"curve.beta2", {0.0005f, pg::BumpScale::absolute}},
        {"curve.beta3", {0.0005f, pg::BumpScale::absolute}},
        {"curve.tau1", {0.005f}},
        {"curve.tau2", {0.005f}},
        {"product.notional", {0.005f}},
        {"product.strike", {0.0005f, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005f}},
    }};
    const auto plan =
        model::prepare_g2_plus_plus_svensson_bermudan_swaption_sensitivities(
            models,
            curves,
            kSwaptions,
            PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            sensitivities,
            {kOrders}
        );
    return benchmark_diagonal(
        "g2_plus_plus_svensson", plan, paths, replay, execution,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, auto selected_replay) {
            return model::
                launch_g2_plus_plus_svensson_bermudan_swaption_diagonal_sensitivities_with_replay_cuda<
                    SwaptionSide::payer, kOrders
                >(
                    host, inputs, stencils, launch, outputs, selected_replay
                );
        },
        [](const auto& host, const auto& launch, auto selected_replay) {
            return model::
                g2_plus_plus_svensson_bermudan_swaption_node_graph_workspace_bytes_with_replay<
                    SwaptionSide::payer, kOrders
                >(host, launch, selected_replay);
        },
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs, void* workspace,
           std::size_t workspace_bytes, auto selected_replay) {
            return model::
                launch_g2_plus_plus_svensson_bermudan_swaption_node_graph_sensitivities_with_replay_cuda<
                    SwaptionSide::payer, kOrders
                >(
                    host, inputs, stencils, launch, outputs,
                    workspace, workspace_bytes, selected_replay
                );
        }
    );
}

int dispatch(
    std::string_view model,
    std::size_t paths,
    lspg::ExerciseReplayStrategy replay,
    ExecutionStrategy execution
) {
    if (model == "black_scholes") {
        return benchmark_black_scholes(paths, replay, execution);
    }
    if (model == "heston") {
        return benchmark_heston(paths, replay, execution);
    }
    if (model == "bates") {
        return benchmark_bates(paths, replay, execution);
    }
    if (model == "cir") {
        return benchmark_cir(paths, replay, execution);
    }
    if (model == "g2") {
        return benchmark_g2(paths, replay, execution);
    }
    if (model == "g2_plus_plus_svensson") {
        return benchmark_g2_plus_plus_svensson(paths, replay, execution);
    }
    throw std::invalid_argument(
        "MODEL must be black_scholes, heston, bates, cir, g2, or "
        "g2_plus_plus_svensson."
    );
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            throw std::invalid_argument(
                "Usage: benchmark_price_gradients_early_exercise_replay "
                "MODEL REPLAY EXECUTION [PATHS]"
            );
        }
        const std::size_t paths = argc == 5
            ? std::stoull(argv[4])
            : 4'096U;
        if (paths == 0U) {
            throw std::invalid_argument("PATHS must be positive.");
        }
        return dispatch(
            argv[1],
            paths,
            parse_replay(argv[2]),
            parse_execution(argv[3])
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
