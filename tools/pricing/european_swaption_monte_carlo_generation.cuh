// Aligned European-swaption MC generation: bounded launches, warmup and publication.
// CUDA allocation/timing and JSON/YAML serialization remain in their shared owners.
#pragma once

#include "common/dataset_validation.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/pricing_launch_plan.hpp"
#include "tools/datasets/price_dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <string>

namespace ai_factory::workbench::datasets {

struct EuropeanSwaptionMonteCarloRecipe {
    std::filesystem::path model_dataset_path;
    std::filesystem::path curve_dataset_path;  // Empty for a standalone model.
    std::filesystem::path product_dataset_path;
    std::filesystem::path dataset_path;
    std::filesystem::path catalog_path;
    std::string url;
    std::uint64_t seed;
    offline::cuda_tuning::PricingIdentity identity{offline::cuda_tuning::PricingFamily::fixed_income_mc, {}, "european_swaption", {}};
    std::size_t paths_per_price = offline::cuda_tuning::kProductionPathsPerPrice;
    unsigned int threads_per_block = offline::cuda_tuning::pricing_profile(identity).threads_per_block;
    std::size_t rows_per_launch = offline::cuda_tuning::pricing_profile(identity).prices_per_launch;
};

template<typename Inputs, typename Launcher>
void generate_european_swaption_monte_carlo_prices(
    const EuropeanSwaptionMonteCarloRecipe& recipe, const Inputs& inputs,
    std::size_t result_count, Launcher&& launcher
) {
    if (recipe.rows_per_launch == 0U || result_count == 0U)
        throw std::invalid_argument("European-swaption generation requires nonempty batches.");
    auto settings = offline::cuda_tuning::pricing_profile(recipe.identity);
    settings.threads_per_block = recipe.threads_per_block;
    settings.prices_per_launch = recipe.rows_per_launch;
    const auto plan = offline::cuda_tuning::make_pricing_launch_plan(
        recipe.identity, result_count, recipe.paths_per_price, settings
    );
    const auto run = offline::cuda::run_monte_carlo(inputs, result_count,
        [&](auto& execution) {
            launcher(execution, 0U, std::min<std::size_t>(4U, plan.prices_per_launch),
                std::min<std::size_t>(4096U, recipe.paths_per_price), plan);
        },
        [&](auto& execution) {
            for (std::size_t offset = 0U; offset < result_count;) {
                const auto count = plan.price_count_at(offset);
                launcher(execution, offset, count, recipe.paths_per_price, plan);
                offset += count;
            }
        }
    );
    const auto publish = [&](const auto&... input_paths) {
      write_monte_carlo_price_dataset(
        input_paths...,
        PriceConstruction::Aligned, run.prices, run.standard_errors, "Philox",
        recipe.dataset_path, recipe.catalog_path, recipe.url,
        "Exact Gaussian joint factor/integral transition + terminal Monte Carlo",
        recipe.paths_per_price, "",
        {{"threads_per_block", recipe.threads_per_block},
         {"block_count", plan.blocks_for(plan.prices_per_launch)},
         {"kernel_launch_count", plan.price_launch_count()},
         {"results_per_launch", plan.prices_per_launch},
         {"launch_plan", offline::cuda_tuning::pricing_launch_metadata(plan)},
         {"work_distribution", "one block per price; independent paths per thread"},
         {"time_day_fraction", "1 / 252"},
         {"tuning_profile", offline::cuda_tuning::metadata("fixed_income_terminal_mc")}},
        {{"pricing_measure", "risk_neutral"},
         {"observation_schedule", {{"transition", "one exact joint transition to exercise"},
             {"discounting", "exact Gaussian integrated rate plus deterministic curve shift when fitted"}}}},
        recipe.seed, run.wall_seconds, run.kernel_seconds
      );
    };
    if (recipe.curve_dataset_path.empty()) {
        publish(recipe.model_dataset_path, recipe.product_dataset_path);
    } else {
        publish(recipe.model_dataset_path, recipe.curve_dataset_path, recipe.product_dataset_path);
    }
    validate_price_dataset_file(recipe.dataset_path);
}

}  // namespace ai_factory::workbench::datasets
