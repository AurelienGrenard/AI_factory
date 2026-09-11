// Host pipeline shared by regular-schedule European-swaption price generators.
#pragma once

#include "product/european_swaption/dataset.hpp"
#include "tools/datasets/price_dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/pricing_launch_plan.hpp"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

inline constexpr float kEuropeanSwaptionTimeDayFraction = 1.0f / 252.0f;

// Price one aligned model/product dataset and serialize its native CUDA output.
template<typename Model, typename Launcher>
void generate_regular_european_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const product::RegularEuropeanSwaptionDataset& product_dataset,
    Launcher launcher,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const std::string& numerical_method,
    const std::string& cuda_label,
    offline::cuda_tuning::PricingIdentity identity
) {
    constexpr PriceConstruction construction = PriceConstruction::Aligned;
    const auto& products = product_dataset.products;
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), construction
    );
    const auto plan = offline::cuda_tuning::make_pricing_launch_plan(
        identity, result_count, 0U
    );
    const auto block_count_for = [&](std::size_t row_count) { return plan.blocks_for(row_count); };
    const std::size_t block_count = block_count_for(result_count);
    (void)cuda_label;
    const auto run = offline::cuda::run_analytical(
        offline::cuda::inputs(models, products),
        result_count,
        [&](auto& execution) {
            const auto* device_models = execution.template input<0U>();
            const auto* device_products = execution.template input<1U>();
            const std::size_t warmup_count = std::min<std::size_t>(
                64U, std::min(models.size(), products.size())
            );
            launcher(
                plan,
                device_models, warmup_count,
                device_products, warmup_count,
                PriceConstruction::Aligned, warmup_count, 0U, warmup_count,
                kEuropeanSwaptionTimeDayFraction,
                plan.profile.threads_per_block,
                block_count_for(warmup_count),
                execution.prices()
            );
        },
        [&](auto& execution) {
            launcher(
                plan,
                execution.template input<0U>(), models.size(),
                execution.template input<1U>(), products.size(),
                construction, result_count, 0U, result_count,
                kEuropeanSwaptionTimeDayFraction,
                plan.profile.threads_per_block,
                block_count,
                execution.prices()
            );
        }
    );

    write_analytical_price_dataset(
        model_dataset_path,
        product_dataset_path,
        construction,
        run.prices,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        nlohmann::ordered_json{
            {"block_count", block_count},
            {"threads_per_block", plan.profile.threads_per_block},
            {"launch_plan", offline::cuda_tuning::pricing_launch_metadata(plan)},
            {"kernel_launch_count", 1U},
            {
                "requested_work_distribution",
                plan.profile.distribution == offline::cuda_tuning::PriceWorkDistribution::block
                    ? "one price per block"
                    : "one price per thread"
            },
        },
        run.wall_seconds,
        run.kernel_seconds
    );
    validate_price_dataset_file(dataset_path);
}

// Price one aligned model/curve/product dataset and serialize its CUDA output.
template<typename Model, typename Curve, typename Launcher>
void generate_regular_european_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& curve_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const std::vector<Curve>& curves,
    const product::RegularEuropeanSwaptionDataset& product_dataset,
    Launcher launcher,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const std::string& numerical_method,
    const std::string& cuda_label,
    offline::cuda_tuning::PricingIdentity identity
) {
    constexpr PriceConstruction construction = PriceConstruction::Aligned;
    const auto& products = product_dataset.products;
    const std::size_t result_count = price_row_count(
        models.size(), curves.size(), products.size(), construction
    );
    const auto plan = offline::cuda_tuning::make_pricing_launch_plan(
        identity, result_count, 0U
    );
    const auto block_count_for = [&](std::size_t row_count) { return plan.blocks_for(row_count); };
    const std::size_t block_count = block_count_for(result_count);
    (void)cuda_label;
    const auto run = offline::cuda::run_analytical(
        offline::cuda::inputs(models, curves, products),
        result_count,
        [&](auto& execution) {
            const std::size_t warmup_count = std::min<std::size_t>(
                64U,
                std::min(models.size(), std::min(curves.size(), products.size()))
            );
            launcher(
                plan,
                execution.template input<0U>(), warmup_count,
                execution.template input<1U>(), warmup_count,
                execution.template input<2U>(), warmup_count,
                PriceConstruction::Aligned, warmup_count, 0U, warmup_count,
                kEuropeanSwaptionTimeDayFraction,
                plan.profile.threads_per_block,
                block_count_for(warmup_count),
                execution.prices()
            );
        },
        [&](auto& execution) {
            launcher(
                plan,
                execution.template input<0U>(), models.size(),
                execution.template input<1U>(), curves.size(),
                execution.template input<2U>(), products.size(),
                construction, result_count, 0U, result_count,
                kEuropeanSwaptionTimeDayFraction,
                plan.profile.threads_per_block,
                block_count,
                execution.prices()
            );
        }
    );

    write_analytical_price_dataset(
        model_dataset_path,
        curve_dataset_path,
        product_dataset_path,
        construction,
        run.prices,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        nlohmann::ordered_json{
            {"block_count", block_count},
            {"threads_per_block", plan.profile.threads_per_block},
            {"launch_plan", offline::cuda_tuning::pricing_launch_metadata(plan)},
            {"kernel_launch_count", 1U},
            {
                "requested_work_distribution",
                plan.profile.distribution == offline::cuda_tuning::PriceWorkDistribution::block
                    ? "one price per block"
                    : "one price per thread"
            },
        },
        run.wall_seconds,
        run.kernel_seconds
    );
    validate_price_dataset_file(dataset_path);
}

}  // namespace ai_factory::workbench::datasets
