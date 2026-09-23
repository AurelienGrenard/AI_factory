// Shared offline gradient execution with native batching, CRN seeds and durable channels.
#pragma once
#include "tools/cuda/price_gradients/launch_plan.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/generation_checkpoint.hpp"
#include "tools/cuda/generation_progress.hpp"
#include "tools/datasets/price_gradients/dataset.hpp"
#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include <iostream>
#include <memory>
#include <type_traits>

namespace ai_factory::workbench::offline::pricing::price_gradients {
namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<typename Plan>
inline constexpr bool has_curve_inputs_v = requires(const Plan& plan) {
    typename Plan::Curve;
    plan.curves;
};

template<typename Plan>
inline constexpr std::size_t stencil_input_index_v =
    has_curve_inputs_v<Plan> ? 4U : 3U;

template<typename Plan>
inline constexpr std::size_t preparation_error_input_index_v =
    has_curve_inputs_v<Plan> ? 5U : 4U;

template<typename Plan, typename Execution>
auto make_device_inputs(const Plan& plan, Execution& execution) {
    if constexpr (has_curve_inputs_v<Plan>) {
        return typename Plan::DeviceInputs{
            execution.template input<0U>(), plan.models.size(),
            execution.template input<1U>(), plan.curves.size(),
            execution.template input<2U>(), plan.products.size(),
            execution.template input<3U>(), plan.sensitivities.size(),
        };
    } else {
        return typename Plan::DeviceInputs{
            execution.template input<0U>(), plan.models.size(),
            execution.template input<1U>(), plan.products.size(),
            execution.template input<2U>(), plan.sensitivities.size(),
        };
    }
}

template<std::size_t NodeCapacity, typename Plan, typename Execution>
auto make_stencil_outputs(
    const Plan&,
    Execution& execution,
    std::size_t stencil_count
) {
    return mcpg::DevicePreparedStencilOutputs<NodeCapacity>{
        execution.template input<stencil_input_index_v<Plan>>(),
        stencil_count,
        execution.template input<preparation_error_input_index_v<Plan>>(),
    };
}

template<typename Plan, typename Stencils, typename PreparationStatus>
auto make_host_inputs(
    const Plan& plan,
    const Stencils& stencils,
    const PreparationStatus& preparation_status
) {
    if constexpr (has_curve_inputs_v<Plan>) {
        return cuda::inputs(
            plan.models,
            plan.curves,
            plan.products,
            plan.sensitivities,
            stencils,
            preparation_status
        );
    } else {
        return cuda::inputs(
            plan.models,
            plan.products,
            plan.sensitivities,
            stencils,
            preparation_status
        );
    }
}

template<
    bool Stochastic,
    pg::SensitivityOrders Orders,
    typename Plan,
    typename Launch,
    typename StencilPreparation>
int execute_prepared_dataset(
    const datasets::price_gradients::Recipe& recipe,
    cuda_tuning::PricingIdentity identity,
    std::uint64_t seed,
    const Plan& prepared,
    Launch launch,
    StencilPreparation prepare_stencils,
    std::size_t paths_per_price
) {
    try {
        const auto start = std::chrono::steady_clock::now();
        const auto rows = prepared.result_count;
        const auto k = prepared.sensitivity_count();
        constexpr bool first_requested = pg::requests_first_v<Orders>;
        constexpr bool second_requested = pg::requests_second_v<Orders>;
        constexpr std::size_t node_capacity =
            pg::SensitivityTraits<Orders>::node_capacity;
        const auto plan = cuda_tuning::make_price_gradient_launch_plan(
            identity, rows, k, Stochastic ? paths_per_price : 0U
        );
        cuda::DeviceBuffer<float> gradients(first_requested ? rows*k : 0U);
        cuda::DeviceBuffer<float> gradient_errors(
            Stochastic && first_requested ? rows*k : 0U
        );
        cuda::DeviceBuffer<float> diagonal_hessians(
            second_requested ? rows*k : 0U
        );
        cuda::DeviceBuffer<float> diagonal_errors(
            Stochastic && second_requested ? rows*k : 0U
        );
        std::vector<std::string> channels{"price"};
        if constexpr (Stochastic) channels.push_back("price_standard_error");
        std::vector<std::size_t> gradient_channels, gradient_error_channels;
        std::vector<std::size_t> diagonal_channels, diagonal_error_channels;
        for (const auto& sensitivity : recipe.configuration.sensitivities) {
            if constexpr (first_requested) {
                gradient_channels.push_back(channels.size());
                channels.push_back("gradient:" + sensitivity.parameter);
                if constexpr (Stochastic) {
                    gradient_error_channels.push_back(channels.size());
                    channels.push_back(
                        "gradient_standard_error:" + sensitivity.parameter
                    );
                }
            }
            if constexpr (second_requested) {
                diagonal_channels.push_back(channels.size());
                channels.push_back("diagonal_hessian:" + sensitivity.parameter);
                if constexpr (Stochastic) {
                    diagonal_error_channels.push_back(channels.size());
                    channels.push_back(
                        "diagonal_hessian_standard_error:"
                        + sensitivity.parameter
                    );
                }
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
        std::vector<pg::SensitivityStencil<node_capacity>>
            represented_stencils(rows*k);
        std::vector<
            equity::price_gradients::device_preparation::Error
        > preparation_status(1U);
        auto invoke = [&](auto& execution, std::size_t offset, std::size_t count) {
            const auto inputs = make_device_inputs(prepared, execution);
            const auto stencil_outputs = make_stencil_outputs<node_capacity>(
                prepared, execution, represented_stencils.size()
            );
            pg::LaunchConfiguration configuration{
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
            float* price_errors = nullptr;
            float* sensitivity_errors = nullptr;
            if constexpr (Stochastic) {
                price_errors = execution.standard_errors();
                sensitivity_errors = gradient_errors.data();
            }
            if constexpr (second_requested) {
                const pg::SensitivityOutputs outputs{
                    execution.prices(), price_errors,
                    gradients.data(), sensitivity_errors,
                    diagonal_hessians.data(),
                    Stochastic ? diagonal_errors.data() : nullptr,
                    rows, rows*k
                };
                launch(
                    prepared, inputs, stencil_outputs, configuration, outputs
                );
            } else {
                const pg::Outputs outputs{
                    execution.prices(), price_errors,
                    gradients.data(), sensitivity_errors, rows, rows*k
                };
                launch(
                    prepared, inputs, stencil_outputs, configuration, outputs
                );
            }
        };
        const auto host_inputs = make_host_inputs(
            prepared, represented_stencils, preparation_status
        );
        auto run = cuda::run<Stochastic>(
            host_inputs,
            rows,
            [&](auto& execution) { invoke(execution, 0U, 1U); },
            [&](auto& execution) {
                if constexpr (Stochastic) {
                if (resumed != 0U && k != 0U) {
                    const auto inputs = make_device_inputs(
                        prepared, execution
                    );
                    const auto stencil_outputs =
                        make_stencil_outputs<node_capacity>(
                            prepared,
                            execution,
                            represented_stencils.size()
                        );
                    prepare_stencils(
                        prepared, inputs, stencil_outputs, 0U, resumed
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
                            durable[0U].data()+offset, offset, count
                        );
                        execution.copy_standard_errors_range_to(
                            durable[1U].data()+offset, offset, count
                        );
                        auto save_selected = [&](const auto& device_values,
                                                 const auto& device_errors,
                                                 const auto& value_channels,
                                                 const auto& error_channels) {
                            std::vector<float> values(count*k), errors(count*k);
                            device_values.copy_range_to(
                                values.data(), offset*k, count*k
                            );
                            device_errors.copy_range_to(
                                errors.data(), offset*k, count*k
                            );
                            for (std::size_t row = 0U; row < count; ++row) {
                                for (std::size_t i = 0U; i < k; ++i) {
                                    durable[value_channels[i]][offset+row] =
                                        values[row*k+i];
                                    durable[error_channels[i]][offset+row] =
                                        errors[row*k+i];
                                }
                            }
                        };
                        if constexpr (first_requested) {
                            save_selected(gradients, gradient_errors,
                                gradient_channels, gradient_error_channels);
                        }
                        if constexpr (second_requested) {
                            save_selected(diagonal_hessians, diagonal_errors,
                                diagonal_channels, diagonal_error_channels);
                        }
                        std::vector<std::span<const float>> slices;
                        for (const auto& channel : durable) {
                            slices.emplace_back(channel.data()+offset, count);
                        }
                        checkpoint->commit(offset, count, slices);
                        progress.record_host_progress(offset+count);
                    } else {
                        progress.record_cuda_progress(offset+count);
                    }
                    } else {
                        progress.record_cuda_progress(offset+count);
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
                                * sizeof(pg::SensitivityStencil<node_capacity>),
                            cudaMemcpyDeviceToHost
                        ),
                        "device-prepared stencil download"
                    );
                }
                check_cuda(
                    cudaMemcpy(
                        preparation_status.data(),
                        execution.template input<
                            preparation_error_input_index_v<Plan>
                        >(),
                        sizeof(preparation_status[0U]),
                        cudaMemcpyDeviceToHost
                    ),
                    "device-preparation status download"
                );
            }
        );
        if (preparation_status[0U].code != 0) {
            throw std::runtime_error(
                "Device sensitivity preparation failed at row "
                + std::to_string(preparation_status[0U].row)
                + ", sensitivity "
                + std::to_string(preparation_status[0U].sensitivity)
                + ", code "
                + std::to_string(preparation_status[0U].code)
            );
        }
        datasets::price_gradients::Results result;
        result.prices = std::move(run.prices);
        if constexpr (first_requested) {
            result.gradients.resize(rows*k);
            gradients.copy_to(result.gradients.data());
        }
        if constexpr (second_requested) {
            result.diagonal_hessians.resize(rows*k);
            diagonal_hessians.copy_to(result.diagonal_hessians.data());
        }
        if constexpr (Stochastic) {
            result.price_errors = std::move(run.standard_errors);
            if constexpr (first_requested) {
                result.gradient_errors.resize(rows*k);
                gradient_errors.copy_to(result.gradient_errors.data());
            }
            if constexpr (second_requested) {
                result.diagonal_hessian_errors.resize(rows*k);
                diagonal_errors.copy_to(
                    result.diagonal_hessian_errors.data()
                );
            }
        if (timer) {
            std::vector<std::span<float>> destinations;
            for (auto& channel : durable) destinations.emplace_back(channel);
            checkpoint->restore_prefix(destinations);
            result.prices = std::move(durable[0U]);
            result.price_errors = std::move(durable[1U]);
            for (std::size_t row = 0U; row < rows; ++row) {
                for (std::size_t i = 0U; i < k; ++i) {
                    if constexpr (first_requested) {
                        result.gradients[row*k+i] =
                            durable[gradient_channels[i]][row];
                        result.gradient_errors[row*k+i] =
                            durable[gradient_error_channels[i]][row];
                    }
                    if constexpr (second_requested) {
                        result.diagonal_hessians[row*k+i] =
                            durable[diagonal_channels[i]][row];
                        result.diagonal_hessian_errors[row*k+i] =
                            durable[diagonal_error_channels[i]][row];
                    }
                }
            }
        }
        }
        result.stencils.reserve(represented_stencils.size());
        for (const auto& stencil : represented_stencils) {
            result.stencils.push_back(pg::legacy_stencil(stencil));
        }
        if constexpr (second_requested) {
            result.diagonal_stencils = std::move(represented_stencils);
        }
        progress.complete();
        result.execution =
            cuda_tuning::device_prepared_price_gradient_launch_metadata(
                plan, k
            );
        result.execution["monte_carlo_paths_per_price"] =
            plan.paths_per_price;
        result.execution["seed"] = seed;
        result.execution["preparation"] = "device_per_row_sensitivity";
        result.execution["scenario_input_bytes"] = 0U;
        result.execution["model_input_bytes"] =
            prepared.models.size()*sizeof(typename Plan::Preparation::Model);
        if constexpr (has_curve_inputs_v<Plan>) {
            result.execution["curve_input_bytes"] =
                prepared.curves.size()
                * sizeof(typename Plan::Preparation::Curve);
        }
        result.execution["product_input_bytes"] =
            prepared.products.size()*sizeof(typename Plan::Preparation::Product);
        result.execution["sensitivity_input_bytes"] =
            prepared.sensitivities.size()*sizeof(typename Plan::SensitivitySpec);
        result.execution["stencil_output_bytes"] =
            rows*k*sizeof(pg::SensitivityStencil<node_capacity>);
        if constexpr (second_requested) {
            result.execution["requested_orders"] = Orders
                == pg::SensitivityOrders::second
                ? nlohmann::ordered_json::array({"diagonal_second"})
                : nlohmann::ordered_json::array({"first", "diagonal_second"});
            result.execution["maximum_live_scenarios"] = k == 0U ? 1U : 4U;
            result.execution["represented_nodes_per_sensitivity"] =
                k == 0U ? 0U : 4U;
        }
        if (timer) {
            result.execution["checkpoint"] = {
                {"schema_version",1},
                {"resumed_prices",resumed},
                {"timing_scope","current process attempt"},
            };
        }
        result.wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start
        ).count();
        result.kernel_seconds = timer ? timer->seconds() : run.kernel_seconds;
        datasets::price_gradients::write_dataset(recipe,result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

template<
    bool Stochastic,
    pg::SensitivityOrders Orders = pg::SensitivityOrders::first,
    typename Models,
    typename Products,
    typename Prepare,
    typename Launch,
    typename StencilPreparation = std::nullptr_t
>
int execute_dataset(const datasets::price_gradients::Recipe& recipe, cuda_tuning::PricingIdentity identity,
                    std::uint64_t seed, const Models& models, const Products& products, Prepare prepare, Launch launch,
                    std::size_t paths_per_price = cuda_tuning::kProductionPathsPerPrice,
                    StencilPreparation prepare_stencils = nullptr) {
    if (recipe.orders != Orders) {
        std::cerr << "Gradient recipe and compiled sensitivity orders differ.\n";
        return 1;
    }
    using Prepared = decltype(prepare(
        models,
        products,
        recipe.construction,
        recipe.time,
        recipe.configuration
    ));
    static_assert(
        Prepared::kDevicePreparedSensitivities,
        "Gradient datasets require compact device-prepared plans."
    );
    static_assert(
        !std::is_same_v<StencilPreparation, std::nullptr_t>,
        "Device-prepared datasets require a stencil preparation callback."
    );
    const Prepared prepared = prepare(
        models,
        products,
        recipe.construction,
        recipe.time,
        recipe.configuration
    );
    return execute_prepared_dataset<Stochastic, Orders>(
        recipe,
        identity,
        seed,
        prepared,
        std::move(launch),
        std::move(prepare_stencils),
        paths_per_price
    );
}

template<
    bool Stochastic,
    pg::SensitivityOrders Orders = pg::SensitivityOrders::first,
    typename Models,
    typename Curves,
    typename Products,
    typename Prepare,
    typename Launch,
    typename StencilPreparation = std::nullptr_t
>
int execute_curve_dataset(
    const datasets::price_gradients::Recipe& recipe,
    cuda_tuning::PricingIdentity identity,
    std::uint64_t seed,
    const Models& models,
    const Curves& curves,
    const Products& products,
    Prepare prepare,
    Launch launch,
    std::size_t paths_per_price = cuda_tuning::kProductionPathsPerPrice,
    StencilPreparation prepare_stencils = nullptr
) {
    if (recipe.orders != Orders) {
        std::cerr
            << "Gradient recipe and compiled sensitivity orders differ.\n";
        return 1;
    }
    using Prepared = decltype(prepare(
        models,
        curves,
        products,
        recipe.construction,
        recipe.time,
        recipe.configuration
    ));
    static_assert(
        Prepared::kDevicePreparedSensitivities,
        "Gradient datasets require compact device-prepared plans."
    );
    static_assert(
        has_curve_inputs_v<Prepared>,
        "Curve datasets require a curve-aware prepared plan."
    );
    static_assert(
        !std::is_same_v<StencilPreparation, std::nullptr_t>,
        "Device-prepared datasets require a stencil preparation callback."
    );
    const Prepared prepared = prepare(
        models,
        curves,
        products,
        recipe.construction,
        recipe.time,
        recipe.configuration
    );
    return execute_prepared_dataset<Stochastic, Orders>(
        recipe,
        identity,
        seed,
        prepared,
        std::move(launch),
        std::move(prepare_stencils),
        paths_per_price
    );
}
}  // namespace ai_factory::workbench::offline::pricing::price_gradients
