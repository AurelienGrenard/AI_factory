// Generic offline assembly for generated equity price recipes.
#pragma once

#include "common/dataset_validation.hpp"
#include "common/price_construction.cuh"
#include "common/result_index.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/datasets/price_dataset.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

struct EquityPriceRecipe {
    std::filesystem::path model_dataset_path;
    std::filesystem::path product_dataset_path;
    std::filesystem::path dataset_path;
    std::filesystem::path catalog_path;
    std::string url;
    std::string numerical_method;
    PriceConstruction construction = PriceConstruction::Aligned;
};

struct BatchedMonteCarloProfile {
    std::size_t paths_per_price = 0U;
    std::size_t results_per_launch = 0U;
    std::size_t block_count_limit = 0U;
    unsigned int threads_per_block = 0U;
    std::uint64_t seed = 0U;
    std::string delta_t_description;
    nlohmann::ordered_json execution_metadata =
        nlohmann::ordered_json::object();
};

struct VolterraMonteCarloProfile {
    std::size_t paths_per_price = 0U;
    float day_fraction = 0.0f;
    float target_dt = 0.0f;
    std::size_t path_chunk_size = 0U;
    std::uint64_t seed = 0U;
    std::string delta_t_description;
};

struct AnalyticalProfile {
    float day_fraction = 0.0f;
    unsigned int threads_per_block = 0U;
    std::uint32_t simulation_steps_per_day = 0U;
};

struct BatchedLaunchContext {
    PriceConstruction construction;
    std::size_t result_count;
    std::size_t result_offset;
    std::size_t launch_result_count;
    std::size_t paths_per_price;
    std::size_t block_count;
    std::uint64_t seed;
};

struct AnalyticalLaunchContext {
    PriceConstruction construction;
    std::size_t result_count;
    std::size_t result_offset;
    std::size_t launch_result_count;
    float day_fraction;
    std::uint32_t simulation_steps_per_day;
    unsigned int threads_per_block;
    std::size_t block_count;
};

struct VolterraLaunchContext {
    PriceConstruction construction;
    std::size_t result_count;
    std::size_t result_index;
    std::size_t paths_per_price;
    float day_fraction;
    float target_dt;
    std::size_t step_count;
    std::size_t path_chunk_size;
    std::uint64_t seed;
};

struct MonteCarloExecution {
    cuda::MonteCarloRun run;
    nlohmann::ordered_json execution_metadata;
};

struct AnalyticalExecution {
    cuda::AnalyticalRun run;
    nlohmann::ordered_json execution_metadata;
};

inline std::size_t warmup_row_count(
    std::size_t model_count,
    std::size_t product_count
) {
    return std::min<std::size_t>(
        64U,
        std::min(model_count, product_count)
    );
}

inline std::uint32_t rounded_volterra_step_count(
    std::uint32_t maturity_days,
    float day_fraction,
    float target_dt
) {
    if (maturity_days == 0U
        || !std::isfinite(day_fraction)
        || !(day_fraction > 0.0f)
        || !std::isfinite(target_dt)
        || !(target_dt > 0.0f)) {
        throw std::invalid_argument(
            "A Volterra price recipe requires positive finite time inputs."
        );
    }
    const double rounded = std::max(
        1.0,
        std::floor(
            static_cast<double>(maturity_days)
                * static_cast<double>(day_fraction)
                / static_cast<double>(target_dt)
            + 0.5
        )
    );
    if (rounded > static_cast<double>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::overflow_error(
            "The Volterra recipe step count exceeds uint32_t."
        );
    }
    return static_cast<std::uint32_t>(rounded);
}

template<class Models, class Products, class Launcher>
MonteCarloExecution execute_batched_monte_carlo(
    const Models& models,
    const Products& products,
    PriceConstruction construction,
    const BatchedMonteCarloProfile& profile,
    Launcher&& launcher
) {
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), construction
    );
    const std::size_t launch_count =
        (result_count + profile.results_per_launch - 1U)
        / profile.results_per_launch;
    const std::size_t maximum_block_count = bounded_block_count(
        std::min(result_count, profile.results_per_launch),
        profile.block_count_limit
    );
    const std::size_t warmup_count = warmup_row_count(
        models.size(), products.size()
    );

    auto run = cuda::run_monte_carlo(
        cuda::inputs(models, products),
        result_count,
        [&](auto& execution) {
            const BatchedLaunchContext context{
                PriceConstruction::Aligned,
                warmup_count,
                0U,
                warmup_count,
                profile.paths_per_price,
                bounded_block_count(
                    warmup_count, profile.block_count_limit
                ),
                profile.seed,
            };
            std::invoke(
                launcher,
                execution.template input<0U>(),
                warmup_count,
                execution.template input<1U>(),
                warmup_count,
                context,
                execution.prices(),
                execution.standard_errors()
            );
        },
        [&](auto& execution) {
            for (std::size_t offset = 0U;
                 offset < result_count;
                 offset += profile.results_per_launch) {
                const std::size_t count = std::min(
                    profile.results_per_launch,
                    result_count - offset
                );
                const BatchedLaunchContext context{
                    construction,
                    result_count,
                    offset,
                    count,
                    profile.paths_per_price,
                    bounded_block_count(count, profile.block_count_limit),
                    profile.seed,
                };
                std::invoke(
                    launcher,
                    execution.template input<0U>(),
                    models.size(),
                    execution.template input<1U>(),
                    products.size(),
                    context,
                    execution.prices(),
                    execution.standard_errors()
                );
            }
        }
    );

    nlohmann::ordered_json metadata = profile.execution_metadata;
    metadata["block_count"] = maximum_block_count;
    metadata["threads_per_block"] = profile.threads_per_block;
    metadata["kernel_launch_count"] = launch_count;
    metadata["maximum_prices_per_block"] = 1U;
    return {std::move(run), std::move(metadata)};
}

template<class Models, class Prepared, class Products, class Launcher>
MonteCarloExecution execute_prepared_batched_monte_carlo(
    const Models& models,
    const Prepared& prepared,
    const Products& products,
    PriceConstruction construction,
    const BatchedMonteCarloProfile& profile,
    Launcher&& launcher
) {
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), construction
    );
    const std::size_t launch_count =
        (result_count + profile.results_per_launch - 1U)
        / profile.results_per_launch;
    const std::size_t maximum_block_count = bounded_block_count(
        std::min(result_count, profile.results_per_launch),
        profile.block_count_limit
    );
    const std::size_t warmup_count = warmup_row_count(
        models.size(), products.size()
    );

    auto run = cuda::run_monte_carlo(
        cuda::inputs(models, prepared, products),
        result_count,
        [&](auto& execution) {
            const BatchedLaunchContext context{
                PriceConstruction::Aligned,
                warmup_count,
                0U,
                warmup_count,
                profile.paths_per_price,
                bounded_block_count(
                    warmup_count, profile.block_count_limit
                ),
                profile.seed,
            };
            std::invoke(
                launcher,
                execution.template input<0U>(),
                warmup_count,
                execution.template input<1U>(),
                warmup_count,
                execution.template input<2U>(),
                warmup_count,
                context,
                execution.prices(),
                execution.standard_errors()
            );
        },
        [&](auto& execution) {
            for (std::size_t offset = 0U;
                 offset < result_count;
                 offset += profile.results_per_launch) {
                const std::size_t count = std::min(
                    profile.results_per_launch,
                    result_count - offset
                );
                const BatchedLaunchContext context{
                    construction,
                    result_count,
                    offset,
                    count,
                    profile.paths_per_price,
                    bounded_block_count(count, profile.block_count_limit),
                    profile.seed,
                };
                std::invoke(
                    launcher,
                    execution.template input<0U>(),
                    models.size(),
                    execution.template input<1U>(),
                    prepared.size(),
                    execution.template input<2U>(),
                    products.size(),
                    context,
                    execution.prices(),
                    execution.standard_errors()
                );
            }
        }
    );

    nlohmann::ordered_json metadata = profile.execution_metadata;
    metadata["block_count"] = maximum_block_count;
    metadata["threads_per_block"] = profile.threads_per_block;
    metadata["kernel_launch_count"] = launch_count;
    metadata["maximum_prices_per_block"] = 1U;
    return {std::move(run), std::move(metadata)};
}

template<class Models, class Products, class WorkspacePlanner, class Launcher>
MonteCarloExecution execute_volterra_monte_carlo(
    const Models& models,
    const Products& products,
    PriceConstruction construction,
    const VolterraMonteCarloProfile& profile,
    WorkspacePlanner&& workspace_planner,
    Launcher&& launcher
) {
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), construction
    );
    std::size_t maximum_step_count = 1U;
    for (const auto& product : products) {
        maximum_step_count = std::max<std::size_t>(
            maximum_step_count,
            rounded_volterra_step_count(
                product.maturity_days,
                profile.day_fraction,
                profile.target_dt
            )
        );
    }
    const auto workspace = std::invoke(
        workspace_planner,
        maximum_step_count,
        profile.paths_per_price,
        profile.path_chunk_size
    );

    auto invoke_row = [&](auto& execution, std::size_t result_index) {
        const ModelProductIndices indices = decode_model_product_result_index(
            result_index,
            products.size(),
            construction
        );
        const VolterraLaunchContext context{
            construction,
            result_count,
            result_index,
            profile.paths_per_price,
            profile.day_fraction,
            profile.target_dt,
            rounded_volterra_step_count(
                products[indices.product_index].maturity_days,
                profile.day_fraction,
                profile.target_dt
            ),
            profile.path_chunk_size,
            profile.seed,
        };
        std::invoke(
            launcher,
            execution.template input<0U>(),
            models.size(),
            execution.template input<1U>(),
            products.size(),
            context,
            execution.workspace(),
            execution.workspace_bytes(),
            execution.prices(),
            execution.standard_errors()
        );
    };

    auto run = cuda::run_monte_carlo_with_workspace(
        cuda::inputs(models, products),
        result_count,
        workspace.workspace_bytes,
        [&](auto& execution) { invoke_row(execution, 0U); },
        [&](auto& execution) {
            for (std::size_t index = 0U; index < result_count; ++index) {
                invoke_row(execution, index);
            }
        }
    );

    return {
        std::move(run),
        nlohmann::ordered_json{
            {"path_chunk_size", profile.path_chunk_size},
            {
                "chunks_per_price",
                (profile.paths_per_price + profile.path_chunk_size - 1U)
                    / profile.path_chunk_size
            },
            {"maximum_step_count", maximum_step_count},
            {"workspace_bytes", workspace.workspace_bytes},
            {"kernel_launch_count", result_count},
        }
    };
}

template<class Models, class Products, class Launcher>
AnalyticalExecution execute_analytical(
    const Models& models,
    const Products& products,
    PriceConstruction construction,
    const AnalyticalProfile& profile,
    Launcher&& launcher
) {
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), construction
    );
    const std::size_t warmup_count = warmup_row_count(
        models.size(), products.size()
    );
    const auto blocks_for = [&](std::size_t count) {
        return (count - 1U) / profile.threads_per_block + 1U;
    };
    auto run = cuda::run_analytical(
        cuda::inputs(models, products),
        result_count,
        [&](auto& execution) {
            const AnalyticalLaunchContext context{
                PriceConstruction::Aligned,
                warmup_count,
                0U,
                warmup_count,
                profile.day_fraction,
                profile.simulation_steps_per_day,
                profile.threads_per_block,
                blocks_for(warmup_count),
            };
            std::invoke(
                launcher,
                execution.template input<0U>(),
                warmup_count,
                execution.template input<1U>(),
                warmup_count,
                context,
                execution.prices()
            );
        },
        [&](auto& execution) {
            const AnalyticalLaunchContext context{
                construction,
                result_count,
                0U,
                result_count,
                profile.day_fraction,
                profile.simulation_steps_per_day,
                profile.threads_per_block,
                blocks_for(result_count),
            };
            std::invoke(
                launcher,
                execution.template input<0U>(),
                models.size(),
                execution.template input<1U>(),
                products.size(),
                context,
                execution.prices()
            );
        }
    );
    nlohmann::ordered_json metadata{
        {"block_count", blocks_for(result_count)},
        {"threads_per_block", profile.threads_per_block},
        {"kernel_launch_count", 1U},
        {"work_distribution", "one price per thread"},
    };
    if (profile.simulation_steps_per_day != 0U) {
        metadata["simulation_steps_per_day"] =
            profile.simulation_steps_per_day;
    }
    return {std::move(run), std::move(metadata)};
}

template<class ModelLoader, class ProductLoader, class Executor>
int generate_monte_carlo_equity_price_dataset(
    const EquityPriceRecipe& recipe,
    const BatchedMonteCarloProfile& profile,
    ModelLoader&& model_loader,
    ProductLoader&& product_loader,
    Executor&& executor
) {
    const auto models = std::invoke(
        model_loader, recipe.model_dataset_path
    );
    const auto products = std::invoke(
        product_loader, recipe.product_dataset_path
    );
    MonteCarloExecution execution = std::invoke(
        executor, models, products, recipe.construction, profile
    );
    datasets::write_monte_carlo_price_dataset(
        recipe.model_dataset_path,
        recipe.product_dataset_path,
        recipe.construction,
        execution.run.prices,
        execution.run.standard_errors,
        "Philox",
        recipe.dataset_path,
        recipe.catalog_path,
        recipe.url,
        recipe.numerical_method,
        profile.paths_per_price,
        profile.delta_t_description,
        execution.execution_metadata,
        nlohmann::ordered_json::object(),
        profile.seed,
        execution.run.wall_seconds,
        execution.run.kernel_seconds
    );
    datasets::validate_price_dataset_file(recipe.dataset_path);
    return 0;
}

template<class ModelLoader, class ProductLoader, class Executor>
int generate_volterra_equity_price_dataset(
    const EquityPriceRecipe& recipe,
    const VolterraMonteCarloProfile& profile,
    ModelLoader&& model_loader,
    ProductLoader&& product_loader,
    Executor&& executor
) {
    const auto models = std::invoke(
        model_loader, recipe.model_dataset_path
    );
    const auto products = std::invoke(
        product_loader, recipe.product_dataset_path
    );
    MonteCarloExecution execution = std::invoke(
        executor, models, products, recipe.construction, profile
    );
    datasets::write_monte_carlo_price_dataset(
        recipe.model_dataset_path,
        recipe.product_dataset_path,
        recipe.construction,
        execution.run.prices,
        execution.run.standard_errors,
        "Philox",
        recipe.dataset_path,
        recipe.catalog_path,
        recipe.url,
        recipe.numerical_method,
        profile.paths_per_price,
        profile.delta_t_description,
        execution.execution_metadata,
        nlohmann::ordered_json::object(),
        profile.seed,
        execution.run.wall_seconds,
        execution.run.kernel_seconds
    );
    datasets::validate_price_dataset_file(recipe.dataset_path);
    return 0;
}

template<class ModelLoader, class ProductLoader, class Executor>
int generate_analytical_equity_price_dataset(
    const EquityPriceRecipe& recipe,
    const AnalyticalProfile& profile,
    ModelLoader&& model_loader,
    ProductLoader&& product_loader,
    Executor&& executor
) {
    const auto models = std::invoke(
        model_loader, recipe.model_dataset_path
    );
    const auto products = std::invoke(
        product_loader, recipe.product_dataset_path
    );
    AnalyticalExecution execution = std::invoke(
        executor, models, products, recipe.construction, profile
    );
    datasets::write_analytical_price_dataset(
        recipe.model_dataset_path,
        recipe.product_dataset_path,
        recipe.construction,
        execution.run.prices,
        recipe.dataset_path,
        recipe.catalog_path,
        recipe.url,
        recipe.numerical_method,
        execution.execution_metadata,
        execution.run.wall_seconds,
        execution.run.kernel_seconds
    );
    datasets::validate_price_dataset_file(recipe.dataset_path);
    return 0;
}

}  // namespace ai_factory::workbench::offline::pricing
