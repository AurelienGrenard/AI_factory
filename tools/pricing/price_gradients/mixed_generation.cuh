// Offline generation of selected mixed Hessians through the device node graph.
#pragma once

#include "tools/pricing/price_gradients/generation.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil_outputs.cuh"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace ai_factory::workbench::offline::pricing::price_gradients {

template<typename Plan>
inline constexpr std::size_t mixed_stencil_input_index_v =
    has_curve_inputs_v<Plan> ? 5U : 4U;

template<typename Plan>
inline constexpr std::size_t mixed_preparation_error_input_index_v =
    has_curve_inputs_v<Plan> ? 6U : 5U;

template<typename Plan, typename Execution>
auto make_mixed_stencil_outputs(
    Execution& execution,
    std::size_t stencil_count
) {
    return pg::MixedSensitivityStencilOutputs{
        execution.template input<mixed_stencil_input_index_v<Plan>>(),
        stencil_count,
    };
}

template<typename Plan, typename Stencils, typename MixedStencils,
         typename PreparationStatus>
auto make_mixed_host_inputs(
    const Plan& plan,
    const Stencils& stencils,
    const MixedStencils& mixed_stencils,
    const PreparationStatus& preparation_status
) {
    if constexpr (has_curve_inputs_v<Plan>) {
        return cuda::inputs(
            plan.models,
            plan.curves,
            plan.products,
            plan.sensitivities,
            stencils,
            mixed_stencils,
            preparation_status
        );
    } else {
        return cuda::inputs(
            plan.models,
            plan.products,
            plan.sensitivities,
            stencils,
            mixed_stencils,
            preparation_status
        );
    }
}

inline bool same_sensitivity_graph(
    const pg::SensitivityGraphPlan& first,
    const pg::SensitivityGraphPlan& second
) {
    return first.first == second.first
        && first.diagonal_second == second.diagonal_second
        && first.mixed_second == second.mixed_second
        && first.coordinate_uses == second.coordinate_uses
        && first.node_capacity == second.node_capacity;
}

inline pg::SensitivityRequest mixed_recipe_request(
    const datasets::price_gradients::Recipe& recipe
) {
    if (!recipe.sensitivity_request.has_value()) {
        throw std::invalid_argument(
            "A mixed-Hessian recipe requires an explicit sensitivity request."
        );
    }
    return *recipe.sensitivity_request;
}

template<
    bool Stochastic,
    typename Plan,
    typename ExecutionPolicy,
    typename StencilPreparation>
int execute_prepared_mixed_dataset(
    const datasets::price_gradients::Recipe& recipe,
    cuda_tuning::PricingIdentity identity,
    std::uint64_t seed,
    const Plan& prepared,
    ExecutionPolicy execution_policy,
    StencilPreparation prepare_stencils,
    std::size_t paths_per_price,
    cuda_tuning::PricingLaunchLimits launch_limits = {}
) {
    try {
        const auto start = std::chrono::steady_clock::now();
        const auto rows = prepared.result_count;
        const auto sensitivity_count = prepared.sensitivity_count();
        const auto& graph = prepared.sensitivity_graph;
        const auto first_count = graph.first.size();
        const auto diagonal_count = graph.diagonal_second.size();
        const auto mixed_count = graph.mixed_second.size();
        if (mixed_count == 0U) {
            throw std::invalid_argument(
                "A mixed node-graph dataset requires a mixed derivative."
            );
        }
        const auto expected_graph = pg::make_sensitivity_graph_plan(
            mixed_recipe_request(recipe), sensitivity_count
        );
        if (!same_sensitivity_graph(graph, expected_graph)) {
            throw std::invalid_argument(
                "Recipe and prepared mixed sensitivity graphs differ."
            );
        }
        const auto axis_stencil_count = checked_workspace_product(
            rows,
            sensitivity_count,
            "Mixed axis-stencil cardinality overflow."
        );
        const auto first_output_count = checked_workspace_product(
            rows,
            first_count,
            "Mixed first-order output cardinality overflow."
        );
        const auto diagonal_output_count = checked_workspace_product(
            rows,
            diagonal_count,
            "Mixed diagonal output cardinality overflow."
        );
        const auto mixed_output_count = checked_workspace_product(
            rows,
            mixed_count,
            "Mixed cross output cardinality overflow."
        );

        const auto plan = cuda_tuning::make_price_gradient_launch_plan(
            identity,
            rows,
            sensitivity_count,
            Stochastic ? paths_per_price : 0U,
            launch_limits
        );
        cuda::DeviceBuffer<float> gradients(first_output_count);
        cuda::DeviceBuffer<float> gradient_errors(
            Stochastic ? first_output_count : 0U
        );
        cuda::DeviceBuffer<float> diagonal_hessians(diagonal_output_count);
        cuda::DeviceBuffer<float> diagonal_errors(
            Stochastic ? diagonal_output_count : 0U
        );
        cuda::DeviceBuffer<float> mixed_hessians(mixed_output_count);
        cuda::DeviceBuffer<float> mixed_errors(
            Stochastic ? mixed_output_count : 0U
        );

        std::vector<std::string> channels{"price"};
        if constexpr (Stochastic) channels.push_back("price_standard_error");
        std::vector<std::size_t> gradient_channels;
        std::vector<std::size_t> gradient_error_channels;
        std::vector<std::size_t> diagonal_channels;
        std::vector<std::size_t> diagonal_error_channels;
        std::vector<std::size_t> mixed_channels;
        std::vector<std::size_t> mixed_error_channels;
        const auto& selections = recipe.configuration.sensitivities;
        const auto add_channel = [&](std::string label) {
            const auto result = channels.size();
            channels.push_back(std::move(label));
            return result;
        };
        for (const auto coordinate : graph.first) {
            const auto& name = selections[coordinate].parameter;
            gradient_channels.push_back(add_channel("gradient:" + name));
            if constexpr (Stochastic) {
                gradient_error_channels.push_back(
                    add_channel("gradient_standard_error:" + name)
                );
            }
        }
        for (const auto coordinate : graph.diagonal_second) {
            const auto& name = selections[coordinate].parameter;
            diagonal_channels.push_back(
                add_channel("diagonal_hessian:" + name)
            );
            if constexpr (Stochastic) {
                diagonal_error_channels.push_back(
                    add_channel("diagonal_hessian_standard_error:" + name)
                );
            }
        }
        for (const auto pair : graph.mixed_second) {
            const auto label = selections[pair.first].parameter + "|"
                + selections[pair.second].parameter;
            mixed_channels.push_back(add_channel("mixed_hessian:" + label));
            if constexpr (Stochastic) {
                mixed_error_channels.push_back(
                    add_channel("mixed_hessian_standard_error:" + label)
                );
            }
        }

        std::unique_ptr<cuda::GenerationCheckpoint> checkpoint;
        std::unique_ptr<cuda::KernelDurationAccumulator> timer;
        if constexpr (Stochastic) {
            checkpoint = std::make_unique<cuda::GenerationCheckpoint>(
                rows, channels
            );
            if (checkpoint->enabled()) {
                timer = std::make_unique<cuda::KernelDurationAccumulator>();
            }
        }
        const auto resumed = checkpoint ? checkpoint->completed_prices() : 0U;
        std::vector<std::vector<float>> durable;
        if (timer) durable.assign(channels.size(), std::vector<float>(rows));
        cuda::GenerationProgress progress(rows, resumed);

        std::vector<pg::SensitivityStencil<4U>> represented_stencils(
            axis_stencil_count
        );
        std::vector<pg::MixedSensitivityStencil> device_mixed_stencils(
            mixed_output_count
        );
        std::vector<
            equity::price_gradients::device_preparation::Error
        > preparation_status(1U);

        const auto make_configuration = [&](
            std::size_t offset, std::size_t count
        ) {
            return pg::LaunchConfiguration{
                Stochastic
                    ? pg::PricingMethod::monte_carlo
                    : pg::PricingMethod::closed_form,
                offset,
                count,
                plan.paths_per_price,
                plan.profile.threads_per_block,
                plan.blocks_for(count),
                seed,
                1U,
            };
        };
        const auto maximum_configuration = make_configuration(
            0U, plan.prices_per_launch
        );
        const auto workspace_bytes = execution_policy.workspace_bytes(
            prepared, maximum_configuration
        );

        auto invoke = [&](
            auto& execution,
            std::size_t offset,
            std::size_t count
        ) {
            const auto inputs = make_device_inputs(prepared, execution);
            const auto stencil_outputs = mcpg::DevicePreparedStencilOutputs<4U>{
                execution.template input<stencil_input_index_v<Plan>>(),
                represented_stencils.size(),
                execution.template input<
                    mixed_preparation_error_input_index_v<Plan>
                >(),
            };
            const auto mixed_stencil_outputs =
                make_mixed_stencil_outputs<Plan>(
                    execution, device_mixed_stencils.size()
                );
            const auto configuration = make_configuration(offset, count);
            float* price_errors = nullptr;
            float* first_errors = nullptr;
            float* second_errors = nullptr;
            float* cross_errors = nullptr;
            if constexpr (Stochastic) {
                price_errors = execution.standard_errors();
                first_errors = gradient_errors.data();
                second_errors = diagonal_errors.data();
                cross_errors = mixed_errors.data();
            }
            const pg::SensitivityOutputs outputs{
                execution.prices(),
                price_errors,
                gradients.data(),
                first_errors,
                diagonal_hessians.data(),
                second_errors,
                rows,
                std::max(first_output_count, diagonal_output_count),
            };
            const pg::MixedSensitivityOutputs mixed_outputs{
                mixed_hessians.data(),
                cross_errors,
                mixed_output_count,
            };
            execution_policy.launch(
                prepared,
                inputs,
                stencil_outputs,
                mixed_stencil_outputs,
                configuration,
                outputs,
                mixed_outputs,
                execution.workspace(),
                execution.workspace_bytes()
            );
        };

        const auto host_inputs = make_mixed_host_inputs(
            prepared,
            represented_stencils,
            device_mixed_stencils,
            preparation_status
        );
        auto run = cuda::run_with_workspace<Stochastic>(
            host_inputs,
            rows,
            workspace_bytes,
            [&](auto& execution) { invoke(execution, 0U, 1U); },
            [&](auto& execution) {
                if constexpr (Stochastic) {
                    if (resumed != 0U && sensitivity_count != 0U) {
                        const auto inputs = make_device_inputs(
                            prepared, execution
                        );
                        const auto stencil_outputs =
                            mcpg::DevicePreparedStencilOutputs<4U>{
                                execution.template input<
                                    stencil_input_index_v<Plan>
                                >(),
                                represented_stencils.size(),
                                execution.template input<
                                    mixed_preparation_error_input_index_v<Plan>
                                >(),
                            };
                        prepare_stencils(
                            prepared,
                            inputs,
                            stencil_outputs,
                            0U,
                            resumed
                        );
                    }
                }
                for (std::size_t offset = resumed; offset < rows;) {
                    const auto count = plan.price_count_at(offset);
                    if (timer) timer->start_batch();
                    invoke(execution, offset, count);
                    if constexpr (Stochastic) {
                        if (timer) {
                            timer->finish_batch();
                            execution.copy_prices_range_to(
                                durable[0U].data() + offset, offset, count
                            );
                            execution.copy_standard_errors_range_to(
                                durable[1U].data() + offset, offset, count
                            );
                            const auto save_selected = [&](
                                const auto& device_values,
                                const auto& device_errors,
                                const auto& value_channels,
                                const auto& error_channels,
                                std::size_t width
                            ) {
                                std::vector<float> values(count * width);
                                std::vector<float> errors(count * width);
                                device_values.copy_range_to(
                                    values.data(), offset * width, count * width
                                );
                                device_errors.copy_range_to(
                                    errors.data(), offset * width, count * width
                                );
                                for (std::size_t row = 0U;
                                     row < count;
                                     ++row) {
                                    for (std::size_t selected = 0U;
                                         selected < width;
                                         ++selected) {
                                        durable[value_channels[selected]][
                                            offset + row
                                        ] = values[row * width + selected];
                                        durable[error_channels[selected]][
                                            offset + row
                                        ] = errors[row * width + selected];
                                    }
                                }
                            };
                            save_selected(
                                gradients,
                                gradient_errors,
                                gradient_channels,
                                gradient_error_channels,
                                first_count
                            );
                            save_selected(
                                diagonal_hessians,
                                diagonal_errors,
                                diagonal_channels,
                                diagonal_error_channels,
                                diagonal_count
                            );
                            save_selected(
                                mixed_hessians,
                                mixed_errors,
                                mixed_channels,
                                mixed_error_channels,
                                mixed_count
                            );
                            std::vector<std::span<const float>> slices;
                            for (const auto& channel : durable) {
                                slices.emplace_back(
                                    channel.data() + offset, count
                                );
                            }
                            checkpoint->commit(offset, count, slices);
                            progress.record_host_progress(offset + count);
                        } else {
                            progress.record_cuda_progress(offset + count);
                        }
                    } else {
                        progress.record_cuda_progress(offset + count);
                    }
                    offset += count;
                }

                if (!represented_stencils.empty()) {
                    check_cuda(
                        cudaMemcpy(
                            represented_stencils.data(),
                            execution.template input<
                                stencil_input_index_v<Plan>
                            >(),
                            represented_stencils.size()
                                * sizeof(pg::SensitivityStencil<4U>),
                            cudaMemcpyDeviceToHost
                        ),
                        "mixed device-prepared stencil download"
                    );
                }
                check_cuda(
                    cudaMemcpy(
                        preparation_status.data(),
                        execution.template input<
                            mixed_preparation_error_input_index_v<Plan>
                        >(),
                        sizeof(preparation_status[0U]),
                        cudaMemcpyDeviceToHost
                    ),
                    "mixed device-preparation status download"
                );
            }
        );

        if (preparation_status[0U].code != 0) {
            throw std::runtime_error(
                "Mixed device sensitivity preparation failed at row "
                + std::to_string(preparation_status[0U].row)
                + ", sensitivity "
                + std::to_string(preparation_status[0U].sensitivity)
                + ", code "
                + std::to_string(preparation_status[0U].code)
            );
        }

        datasets::price_gradients::Results result;
        result.prices = std::move(run.prices);
        result.gradients.resize(first_output_count);
        result.diagonal_hessians.resize(diagonal_output_count);
        result.mixed_hessians.resize(mixed_output_count);
        gradients.copy_to(result.gradients.data());
        diagonal_hessians.copy_to(result.diagonal_hessians.data());
        mixed_hessians.copy_to(result.mixed_hessians.data());
        if constexpr (Stochastic) {
            result.price_errors = std::move(run.standard_errors);
            result.gradient_errors.resize(first_output_count);
            result.diagonal_hessian_errors.resize(diagonal_output_count);
            result.mixed_hessian_errors.resize(mixed_output_count);
            gradient_errors.copy_to(result.gradient_errors.data());
            diagonal_errors.copy_to(result.diagonal_hessian_errors.data());
            mixed_errors.copy_to(result.mixed_hessian_errors.data());
            if (timer) {
                std::vector<std::span<float>> destinations;
                for (auto& channel : durable) {
                    destinations.emplace_back(channel);
                }
                checkpoint->restore_prefix(destinations);
                result.prices = std::move(durable[0U]);
                result.price_errors = std::move(durable[1U]);
                for (std::size_t row = 0U; row < rows; ++row) {
                    for (std::size_t selected = 0U;
                         selected < first_count;
                         ++selected) {
                        result.gradients[row * first_count + selected] =
                            durable[gradient_channels[selected]][row];
                        result.gradient_errors[row * first_count + selected] =
                            durable[gradient_error_channels[selected]][row];
                    }
                    for (std::size_t selected = 0U;
                         selected < diagonal_count;
                         ++selected) {
                        result.diagonal_hessians[
                            row * diagonal_count + selected
                        ] = durable[diagonal_channels[selected]][row];
                        result.diagonal_hessian_errors[
                            row * diagonal_count + selected
                        ] = durable[diagonal_error_channels[selected]][row];
                    }
                    for (std::size_t selected = 0U;
                         selected < mixed_count;
                         ++selected) {
                        result.mixed_hessians[row * mixed_count + selected] =
                            durable[mixed_channels[selected]][row];
                        result.mixed_hessian_errors[
                            row * mixed_count + selected
                        ] = durable[mixed_error_channels[selected]][row];
                    }
                }
            }
        }

        result.stencils.reserve(represented_stencils.size());
        for (const auto& stencil : represented_stencils) {
            result.stencils.push_back(pg::legacy_stencil(stencil));
        }
        if (diagonal_count != 0U) {
            result.diagonal_stencils = represented_stencils;
        }
        result.mixed_stencils.reserve(mixed_output_count);
        for (std::size_t row = 0U; row < rows; ++row) {
            for (const auto pair : graph.mixed_second) {
                result.mixed_stencils.push_back(
                    pg::make_mixed_sensitivity_stencil(
                        represented_stencils[
                            row * sensitivity_count + pair.first
                        ],
                        represented_stencils[
                            row * sensitivity_count + pair.second
                        ]
                    )
                );
            }
        }
        result.sensitivity_graph = graph;

        progress.complete();
        result.execution = cuda_tuning::mixed_node_graph_launch_metadata(
            plan, sensitivity_count, graph
        );
        result.execution["monte_carlo_paths_per_price"] =
            plan.paths_per_price;
        result.execution["seed"] = seed;
        result.execution["preparation"] = "device_per_row_sensitivity_graph";
        datasets::price_gradients::append_exercise_replay_metadata(
            recipe, result.execution
        );
        result.execution["model_input_bytes"] =
            prepared.models.size() * sizeof(typename Plan::Preparation::Model);
        if constexpr (has_curve_inputs_v<Plan>) {
            result.execution["curve_input_bytes"] =
                prepared.curves.size()
                * sizeof(typename Plan::Preparation::Curve);
        }
        result.execution["product_input_bytes"] =
            prepared.products.size()
            * sizeof(typename Plan::Preparation::Product);
        result.execution["sensitivity_input_bytes"] =
            prepared.sensitivities.size()
            * sizeof(typename Plan::SensitivitySpec);
        const auto axis_stencil_bytes = checked_workspace_product(
            axis_stencil_count,
            sizeof(pg::SensitivityStencil<4U>),
            "Mixed axis-stencil byte count overflow."
        );
        const auto mixed_stencil_bytes = checked_workspace_product(
            mixed_output_count,
            sizeof(pg::MixedSensitivityStencil),
            "Mixed cross-stencil byte count overflow."
        );
        if (mixed_stencil_bytes
            > std::numeric_limits<std::size_t>::max()
                - axis_stencil_bytes) {
            throw std::overflow_error(
                "Mixed stencil output byte count overflow."
            );
        }
        result.execution["stencil_output_bytes"] =
            axis_stencil_bytes + mixed_stencil_bytes;
        result.execution["sensitivity_workspace_bytes"] = workspace_bytes;
        result.execution["sensitivity_strategy"] =
            execution_policy.strategy_name();
        if (timer) {
            result.execution["checkpoint"] = {
                {"schema_version", 1},
                {"resumed_prices", resumed},
                {"timing_scope", "current process attempt"},
            };
        }
        result.wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();
        result.kernel_seconds = timer ? timer->seconds() : run.kernel_seconds;
        datasets::price_gradients::write_dataset(recipe, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

template<
    bool Stochastic,
    typename Models,
    typename Products,
    typename Prepare,
    typename GraphWorkspaceBytes,
    typename GraphLaunch,
    typename StencilPreparation>
int execute_mixed_node_graph_dataset(
    const datasets::price_gradients::Recipe& recipe,
    cuda_tuning::PricingIdentity identity,
    std::uint64_t seed,
    const Models& models,
    const Products& products,
    Prepare prepare,
    GraphWorkspaceBytes graph_workspace_bytes,
    GraphLaunch graph_launch,
    std::size_t paths_per_price,
    StencilPreparation prepare_stencils,
    cuda_tuning::PricingLaunchLimits launch_limits = {}
) {
    const auto request = mixed_recipe_request(recipe);
    using Prepared = decltype(prepare(
        models,
        products,
        recipe.construction,
        recipe.time,
        recipe.configuration,
        request
    ));
    static_assert(
        Prepared::kDevicePreparedSensitivities,
        "Mixed datasets require compact device-prepared plans."
    );
    const Prepared prepared = prepare(
        models,
        products,
        recipe.construction,
        recipe.time,
        recipe.configuration,
        request
    );
    return execute_prepared_mixed_dataset<Stochastic>(
        recipe,
        identity,
        seed,
        prepared,
        mixed_node_graph_sensitivity_execution(
            std::move(graph_workspace_bytes),
            std::move(graph_launch)
        ),
        std::move(prepare_stencils),
        paths_per_price,
        launch_limits
    );
}

template<
    bool Stochastic,
    typename Models,
    typename Curves,
    typename Products,
    typename Prepare,
    typename GraphWorkspaceBytes,
    typename GraphLaunch,
    typename StencilPreparation>
int execute_curve_mixed_node_graph_dataset(
    const datasets::price_gradients::Recipe& recipe,
    cuda_tuning::PricingIdentity identity,
    std::uint64_t seed,
    const Models& models,
    const Curves& curves,
    const Products& products,
    Prepare prepare,
    GraphWorkspaceBytes graph_workspace_bytes,
    GraphLaunch graph_launch,
    std::size_t paths_per_price,
    StencilPreparation prepare_stencils,
    cuda_tuning::PricingLaunchLimits launch_limits = {}
) {
    const auto request = mixed_recipe_request(recipe);
    using Prepared = decltype(prepare(
        models,
        curves,
        products,
        recipe.construction,
        recipe.time,
        recipe.configuration,
        request
    ));
    static_assert(
        Prepared::kDevicePreparedSensitivities,
        "Mixed datasets require compact device-prepared plans."
    );
    static_assert(
        has_curve_inputs_v<Prepared>,
        "Curve mixed datasets require a curve-aware prepared plan."
    );
    const Prepared prepared = prepare(
        models,
        curves,
        products,
        recipe.construction,
        recipe.time,
        recipe.configuration,
        request
    );
    return execute_prepared_mixed_dataset<Stochastic>(
        recipe,
        identity,
        seed,
        prepared,
        mixed_node_graph_sensitivity_execution(
            std::move(graph_workspace_bytes),
            std::move(graph_launch)
        ),
        std::move(prepare_stencils),
        paths_per_price,
        launch_limits
    );
}

}  // namespace ai_factory::workbench::offline::pricing::price_gradients
