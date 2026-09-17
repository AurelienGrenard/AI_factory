// Shared offline gradient execution with native batching, CRN seeds and durable channels.
#pragma once
#include "tools/cuda/price_gradients/launch_plan.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/generation_checkpoint.hpp"
#include "tools/cuda/generation_progress.hpp"
#include "tools/datasets/price_gradients/dataset.hpp"
#include "common/price_gradients/launch.cuh"
#include <iostream>
#include <memory>

namespace ai_factory::workbench::offline::pricing::price_gradients {
namespace pg = ::ai_factory::workbench::price_gradients;

template<bool Stochastic, typename Models, typename Products, typename Prepare, typename Launch>
int execute_dataset(const datasets::price_gradients::Recipe& recipe, cuda_tuning::PricingIdentity identity,
                    std::uint64_t seed, const Models& models, const Products& products, Prepare prepare, Launch launch,
                    std::size_t paths_per_price = cuda_tuning::kProductionPathsPerPrice) {
    try {
        const auto start = std::chrono::steady_clock::now();
        const auto scenarios = prepare(models, products, recipe.construction, recipe.time, recipe.configuration);
        using Scenario = typename decltype(scenarios)::ScenarioType;
        const auto rows = scenarios.result_count, k = scenarios.sensitivity_count();
        const auto plan = cuda_tuning::make_price_gradient_launch_plan(identity, rows, k,
            Stochastic ? paths_per_price : 0U);
        cuda::DeviceBuffer<float> gradients(rows*k), gradient_errors(Stochastic ? rows*k : 0U);
        std::vector<std::string> channels{"price", "price_standard_error"};
        for (const auto& sensitivity : recipe.configuration.sensitivities) {
            channels.push_back("gradient:" + sensitivity.parameter);
            channels.push_back("gradient_standard_error:" + sensitivity.parameter);
        }
        std::unique_ptr<cuda::GenerationCheckpoint> checkpoint;
        std::unique_ptr<cuda::KernelDurationAccumulator> timer;
        if constexpr (Stochastic) {
            checkpoint = std::make_unique<cuda::GenerationCheckpoint>(rows, channels);
            if (checkpoint->enabled()) timer = std::make_unique<cuda::KernelDurationAccumulator>();
        }
        const auto resumed = checkpoint ? checkpoint->completed_prices() : 0U;
        std::vector<std::vector<float>> durable;
        if (timer) durable.assign(channels.size(), std::vector<float>(rows));
        cuda::GenerationProgress progress(rows, resumed);
        auto invoke = [&](auto& execution, std::size_t offset, std::size_t count) {
            pg::DeviceInputs<Scenario> inputs{execution.template input<0>(), scenarios.scenarios.size(),
                execution.template input<1>(), scenarios.stencils.size()};
            pg::LaunchConfiguration configuration{Stochastic ? pg::PricingMethod::monte_carlo : pg::PricingMethod::closed_form,
                offset, count, plan.paths_per_price, plan.profile.threads_per_block, plan.blocks_for(count), seed, plan.sensitivity_batch_size};
            float* errors = nullptr;
            if constexpr (Stochastic) errors = execution.standard_errors();
            pg::Outputs outputs{execution.prices(), errors, gradients.data(), gradient_errors.data(), rows, rows*k};
            launch(scenarios, inputs, configuration, outputs);
        };
        auto run = cuda::run<Stochastic>(cuda::inputs(scenarios.scenarios, scenarios.stencils), rows,
            [&](auto& execution) { invoke(execution, 0U, 1U); },
            [&](auto& execution) {
                for (std::size_t offset = resumed; offset < rows;) {
                    const auto count = plan.price_count_at(offset);
                    if (timer) timer->start_batch();
                    invoke(execution, offset, count);
                    if constexpr (Stochastic) {
                        if (timer) {
                            timer->finish_batch();
                            execution.copy_prices_range_to(durable[0].data()+offset, offset, count);
                            execution.copy_standard_errors_range_to(durable[1].data()+offset, offset, count);
                            std::vector<float> values(count*k), errors(count*k);
                            gradients.copy_range_to(values.data(), offset*k, count*k);
                            gradient_errors.copy_range_to(errors.data(), offset*k, count*k);
                            for (std::size_t row = 0; row < count; ++row) for (std::size_t i = 0; i < k; ++i) {
                                durable[2U+2U*i][offset+row] = values[row*k+i];
                                durable[3U+2U*i][offset+row] = errors[row*k+i];
                            }
                            std::vector<std::span<const float>> slices;
                            for (const auto& channel : durable) slices.emplace_back(channel.data()+offset, count);
                            checkpoint->commit(offset,count,slices);
                            progress.record_host_progress(offset+count);
                        } else progress.record_cuda_progress(offset+count);
                    } else progress.record_cuda_progress(offset+count);
                    offset += count;
                }
            });
        datasets::price_gradients::Results result;
        result.prices = std::move(run.prices);
        result.gradients.resize(rows*k);
        gradients.copy_to(result.gradients.data());
        if constexpr (Stochastic) {
            result.price_errors = std::move(run.standard_errors);
            result.gradient_errors.resize(rows*k);
            gradient_errors.copy_to(result.gradient_errors.data());
            if (timer) {
                std::vector<std::span<float>> destinations;
                for (auto& channel : durable) destinations.emplace_back(channel);
                checkpoint->restore_prefix(destinations);
                result.prices = std::move(durable[0]);
                result.price_errors = std::move(durable[1]);
                for (std::size_t row = 0; row < rows; ++row) for (std::size_t i = 0; i < k; ++i) {
                    result.gradients[row*k+i] = durable[2U+2U*i][row];
                    result.gradient_errors[row*k+i] = durable[3U+2U*i][row];
                }
            }
        }
        progress.complete();
        result.stencils = scenarios.stencils;
        result.execution = cuda_tuning::price_gradient_launch_metadata(plan,k);
        result.execution["monte_carlo_paths_per_price"] = plan.paths_per_price;
        result.execution["seed"] = seed;
        result.execution["scenario_input_bytes"] = scenarios.scenarios.size()*sizeof(Scenario);
        result.execution["stencil_input_bytes"] = scenarios.stencils.size()*sizeof(pg::Stencil);
        if (timer) result.execution["checkpoint"] = {{"schema_version",1},{"resumed_prices",resumed},{"timing_scope","current process attempt"}};
        result.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        result.kernel_seconds = timer ? timer->seconds() : run.kernel_seconds;
        datasets::price_gradients::write_dataset(recipe,result);
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
}  // namespace ai_factory::workbench::offline::pricing::price_gradients
