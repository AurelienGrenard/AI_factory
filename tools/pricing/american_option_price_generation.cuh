// Shared host pipeline for American-option price datasets.
#pragma once

#include "common/longstaff_schwartz/launch.cuh"
#include "product/american_option/dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/tuning_profile.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace ai_factory::workbench::offline::pricing {

struct AmericanOptionProfile {
    std::size_t paths_per_price = 0U;
    unsigned int threads_per_block = 0U;
    std::size_t blocks_per_price = 0U;
    std::uint64_t seed = 0U;
    std::string delta_t_description;
    std::string diagnostic_label;
    std::string regression_basis;
    std::string regression_precision;
    nlohmann::ordered_json time_discretization =
        nlohmann::ordered_json::object();
    nlohmann::ordered_json basis_state = nlohmann::ordered_json::array();
    nlohmann::ordered_json basis_normalization =
        nlohmann::ordered_json::array();
    nlohmann::ordered_json basis_functions = nlohmann::ordered_json::array();
    bool exact_exercise_dates = false;
};

inline nlohmann::ordered_json american_option_catalog_sections(
    const AmericanOptionProfile& profile
) {
    nlohmann::ordered_json sections = {
        {"outputs", {
            {
                "price",
                {{"estimator", "Longstaff-Schwartz discounted cashflow mean"}}
            },
            {
                "standard_error",
                {{
                    "estimator",
                    "Conditional standard error of discounted policy cashflows"
                }}
            },
        }},
        {"exercise_policy", {
            {"method", "Longstaff-Schwartz"},
            {"regression_basis", profile.regression_basis},
            {"exercise_dates", "time zero and maturity-anchored product dates"},
            {
                "regression_target",
                "realized future cashflow discounted one exercise interval"
            },
            {"in_the_money_paths_only", true},
            {"solver", profile.regression_precision},
            {"failure_policy", "fatal regression statuses reject publication"},
            {"basis", {
                {"state", profile.basis_state},
                {"normalization", profile.basis_normalization},
                {"functions", profile.basis_functions},
                {
                    "regularization",
                    {{"ridge", "1e-10 * trace(G) / basis_size"}}
                },
            }},
        }},
    };
    if (profile.exact_exercise_dates) {
        sections["observation_schedule"] = {
            {"rule", "exact independent increments on exercise dates"},
            {"initial_stub", {
                {"increments", 1U},
                {"interval", "first_exercise_time"},
            }},
            {"regular_exercise_interval", {
                {"increments", 1U},
                {"interval", "exercise_interval"},
            }},
        };
    }
    return sections;
}

inline nlohmann::ordered_json american_option_cuda_execution(
    const AmericanOptionProfile& profile,
    const longstaff_schwartz::LaunchResult& execution
) {
    nlohmann::ordered_json metadata = {
        {"threads_per_block", profile.threads_per_block},
        {"blocks_per_price", execution.blocks_per_price},
        {"batch_count", execution.batch_count},
        {"kernel_launch_count", execution.kernel_launch_count},
        {"workspace_bytes", execution.workspace_bytes},
        {"tuning_profile", cuda_tuning::metadata("equity_early_exercise")},
    };
    for (const auto& [name, value] : profile.time_discretization.items()) {
        metadata[name] = value;
    }
    return metadata;
}

template<class ModelLoader, class Launcher>
int generate_american_option_equity_price_dataset(
    const EquityPriceRecipe& recipe,
    const AmericanOptionProfile& profile,
    ModelLoader&& model_loader,
    Launcher&& launcher
) {
    const auto models = std::invoke(
        std::forward<ModelLoader>(model_loader), recipe.model_dataset_path
    );
    const auto products = product::load_american_options(
        recipe.product_dataset_path
    );
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), recipe.construction
    );
    longstaff_schwartz::LaunchResult execution{};
    const auto run = cuda::run_monte_carlo(
        cuda::inputs(models, products),
        result_count,
        [&](auto& resources) {
            std::invoke(
                launcher,
                resources.template input<0U>(),
                1U,
                products.data(),
                resources.template input<1U>(),
                1U,
                PriceConstruction::Aligned,
                1U,
                std::min<std::size_t>(profile.paths_per_price, 4'096U),
                resources.prices(),
                resources.standard_errors()
            );
        },
        [&](auto& resources) {
            execution = std::invoke(
                launcher,
                resources.template input<0U>(),
                models.size(),
                products.data(),
                resources.template input<1U>(),
                products.size(),
                recipe.construction,
                result_count,
                profile.paths_per_price,
                resources.prices(),
                resources.standard_errors()
            );
            longstaff_schwartz::validate_regression_diagnostics(
                execution, profile.diagnostic_label.c_str()
            );
        }
    );

    datasets::write_monte_carlo_price_dataset(
        recipe.model_dataset_path,
        recipe.product_dataset_path,
        recipe.construction,
        run.prices,
        run.standard_errors,
        "Philox",
        recipe.dataset_path,
        recipe.catalog_path,
        recipe.url,
        recipe.numerical_method,
        profile.paths_per_price,
        profile.delta_t_description,
        american_option_cuda_execution(profile, execution),
        american_option_catalog_sections(profile),
        profile.seed,
        run.wall_seconds,
        execution.kernel_seconds
    );
    datasets::validate_price_dataset_file(recipe.dataset_path);
    return 0;
}

}  // namespace ai_factory::workbench::offline::pricing
