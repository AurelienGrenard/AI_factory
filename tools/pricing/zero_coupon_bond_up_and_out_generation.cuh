// Grid-monitored fixed-income bond barrier generation with explicit MC provenance.
#pragma once

#include "common/dataset_validation.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/pricing_launch_plan.hpp"
#include "tools/datasets/price_dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace ai_factory::workbench::datasets {

struct ZeroCouponBondUpAndOutRecipe {
    std::filesystem::path model_dataset_path;
    std::filesystem::path product_dataset_path;
    std::filesystem::path dataset_path;
    std::filesystem::path catalog_path;
    std::string url;
    std::uint64_t seed;
    offline::cuda_tuning::PricingIdentity identity{
        offline::cuda_tuning::PricingFamily::fixed_income_mc,
        "ornstein_uhlenbeck", "zero_coupon_bond_up_and_out", ""};
    PriceConstruction construction = PriceConstruction::Aligned;
    std::size_t paths_per_price = 1'048'576U;
    std::uint32_t simulation_steps_per_day = 1U;
    std::filesystem::path curve_dataset_path{};
    std::string transition_method =
        "exact joint Gaussian factor and stochastic rate-integral transitions";
};

template<typename Inputs, typename Launcher>
void generate_zero_coupon_bond_up_and_out_prices(
    const ZeroCouponBondUpAndOutRecipe& recipe,
    const Inputs& inputs,
    std::size_t result_count,
    Launcher&& launcher
) {
    if (result_count == 0U || recipe.simulation_steps_per_day == 0U)
        throw std::invalid_argument("Barrier price generation requires a nonempty grid.");
    const auto plan = offline::cuda_tuning::make_pricing_launch_plan(
        recipe.identity, result_count, recipe.paths_per_price,
        offline::cuda_tuning::pricing_profile(recipe.identity));
    auto run = offline::cuda::run_monte_carlo(inputs, result_count,
        [&](auto& execution) {
            launcher(execution, 0U, std::min<std::size_t>(4U, result_count),
                     std::min<std::size_t>(4096U, recipe.paths_per_price), plan);
        },
        [&](auto& execution) {
            for (std::size_t offset = 0U; offset < result_count;) {
                const auto count = plan.price_count_at(offset);
                launcher(execution, offset, count, recipe.paths_per_price, plan);
                offset += count;
            }
        });
    const nlohmann::ordered_json execution_metadata{
        {"threads_per_block", plan.profile.threads_per_block},
        {"block_count", plan.blocks_for(plan.prices_per_launch)},
        {"kernel_launch_count", plan.price_launch_count()},
        {"results_per_launch", plan.prices_per_launch},
        {"launch_plan", offline::cuda_tuning::pricing_launch_metadata(plan)},
        {"work_distribution", "one block per price; independent paths per thread"},
        {"time_day_fraction", "1 / 252"},
        {"simulation_steps_per_day", recipe.simulation_steps_per_day},
        {"time_grid", "dt = 1 / (252 * simulation_steps_per_day)"},
    };
    const nlohmann::ordered_json contract_metadata{
        {"pricing_measure", "risk_neutral"},
        {"barrier_monitoring", "discrete at t=0 and every numerical grid point through T; continuous-barrier price approximated from above"},
        {"barrier_breach", "P(t,U) >= B"},
        {"discounting", recipe.transition_method},
    };
    const auto write = [&](auto&&... inputs) {
        write_monte_carlo_price_dataset(
            recipe.model_dataset_path, std::forward<decltype(inputs)>(inputs)...,
            recipe.product_dataset_path, recipe.construction,
            run.prices, run.standard_errors, "Philox", recipe.dataset_path,
            recipe.catalog_path, recipe.url,
            recipe.transition_method + "; grid-observed bond barrier",
            recipe.paths_per_price, "1 / (252 * simulation_steps_per_day)",
            execution_metadata, contract_metadata, recipe.seed,
            run.wall_seconds, run.kernel_seconds);
    };
    if (recipe.curve_dataset_path.empty()) write();
    else write(recipe.curve_dataset_path);
    validate_price_dataset_file(recipe.dataset_path);
}

}  // namespace ai_factory::workbench::datasets
