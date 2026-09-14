// Native paired-output recipes: production launch plan, RAII execution and publication.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "common/result_index.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/pricing_launch_plan.hpp"
#include "tools/cuda/generation_progress.hpp"
#include "tools/datasets/price_delta_dataset.hpp"
#include <iostream>

namespace ai_factory::workbench::offline::pricing {

struct PriceDeltaLaunchContext {
    std::size_t results, offset, count, paths, blocks;
    unsigned threads;
    std::uint64_t seed;
    equity::price_delta::SpotBumpConfiguration bump;
};

template<bool Stochastic, bool EarlyExercise, typename Models, typename Products, typename Launcher>
int execute_equity_price_delta_dataset(const datasets::PriceDeltaRecipe& recipe,
    cuda_tuning::PricingIdentity identity, std::uint64_t seed,
    const Models& models, const Products& products, Launcher launch,
    std::chrono::steady_clock::time_point wall_start,
    nlohmann::ordered_json preparation = nlohmann::ordered_json::object()) {
    try {
        const auto result_count = price_row_count(
            models.size(), products.size(), recipe.construction);
        const auto plan = cuda_tuning::make_equity_price_delta_launch_plan(identity, result_count,
            Stochastic ? cuda_tuning::kProductionPathsPerPrice : 0U);
        cuda::GenerationProgress progress(result_count);
        const equity::price_delta::SpotBumpConfiguration bump{static_cast<float>(recipe.relative_bump_width)};
        datasets::PriceDeltaResults result;
        for (std::size_t result_index = 0; result_index < result_count; ++result_index) {
            const auto indices = decode_model_product_result_index(
                result_index, products.size(), recipe.construction);
            const auto& model = models.at(indices.model_index);
            equity::price_delta::validate_spot_bump(model.spot, bump);
            const auto endpoints = equity::price_delta::prepare_spot_bump(model.spot, bump);
            result.lower_spots.push_back(endpoints.lower);
            result.upper_spots.push_back(endpoints.upper);
            result.bump_widths.push_back(endpoints.width);
        }
        cuda::DeviceBuffer<float> deltas(result_count), errors(Stochastic ? result_count : 0U);
        longstaff_schwartz::LaunchResult lsm_result{};
        auto invoke = [&](auto& execution, std::size_t model_count,
                          std::size_t product_count, std::size_t results,
                          std::size_t offset, std::size_t count) {
            const PriceDeltaLaunchContext context{results, offset, count, plan.paths_per_price,
                plan.profile.distribution == cuda_tuning::PriceWorkDistribution::fft ? 0U : plan.blocks_for(count), plan.profile.threads_per_block, seed, bump};
            float* price_errors = nullptr;
            if constexpr (Stochastic) price_errors = execution.standard_errors();
            if constexpr (EarlyExercise) {
                lsm_result = launch(models.data(), execution.template input<0>(), model_count,
                    products.data(), execution.template input<1>(), product_count, context,
                    execution.prices(), price_errors, deltas.data(), errors.data());
                longstaff_schwartz::validate_regression_diagnostics(lsm_result, "American price-delta");
            } else {
                launch(models.data(), execution.template input<0>(), model_count,
                    products.data(), execution.template input<1>(), product_count, context,
                    execution.prices(), price_errors, deltas.data(), errors.data());
            }
        };
        const auto run = cuda::run<Stochastic>(cuda::inputs(models, products), result_count,
            [&](auto& execution) {
                const auto warmup = std::min<std::size_t>(1U, plan.prices_per_launch);
                invoke(execution, warmup, warmup, warmup, 0, warmup);
            },
            [&](auto& execution) {
                if constexpr (EarlyExercise) {
                    cuda::ScopedGenerationProgress active(progress);
                    invoke(execution, models.size(), products.size(),
                        result_count, 0, result_count);
                }
                else for (std::size_t offset = 0; offset < result_count;) {
                    const auto count = plan.price_count_at(offset);
                    invoke(execution, models.size(), products.size(),
                        result_count, offset, count);
                    offset += count;
                    progress.record_cuda_progress(offset);
                }
            });
        progress.complete();
        result.prices = run.prices;
        if constexpr (Stochastic) result.price_errors = run.standard_errors;
        result.deltas.resize(result_count);
        deltas.copy_to(result.deltas.data());
        if constexpr (Stochastic) {
            result.delta_errors.resize(result_count);
            errors.copy_to(result.delta_errors.data());
        }
        result.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        result.kernel_seconds = EarlyExercise ? lsm_result.kernel_seconds : run.kernel_seconds;
        result.execution = cuda_tuning::pricing_launch_metadata(plan);
        result.execution["monte_carlo_paths_per_price"] = plan.paths_per_price;
        result.execution["seed"] = seed;
        if (!preparation.empty()) result.execution["preparation"] = std::move(preparation);
        if constexpr (EarlyExercise) {
            result.execution["kernel_launch_count"] = lsm_result.kernel_launch_count;
            result.execution["batch_count"] = lsm_result.batch_count;
            result.execution["maximum_prices_per_batch"] = lsm_result.maximum_prices_per_batch;
            result.execution["workspace_bytes"] = lsm_result.workspace_bytes;
        }
        datasets::write_price_delta_dataset(recipe, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

template<bool Stochastic, bool EarlyExercise, typename ModelLoader, typename ProductLoader,
         typename Launcher>
int generate_equity_price_delta_dataset(const datasets::PriceDeltaRecipe& recipe,
    cuda_tuning::PricingIdentity identity, std::uint64_t seed,
    ModelLoader load_models, ProductLoader load_products, Launcher launch) {
    try {
        const auto models = load_models(recipe.model_input);
        const auto products = load_products(recipe.product_input);
        return execute_equity_price_delta_dataset<Stochastic, EarlyExercise>(
            recipe, identity, seed, models, products, launch, std::chrono::steady_clock::now());
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

}  // namespace ai_factory::workbench::offline::pricing
