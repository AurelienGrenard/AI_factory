// Host pipeline shared by regular-schedule European-swaption price generators.
#pragma once

#include "product/european_swaption/dataset.hpp"
#include "tools/datasets/price_dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

inline constexpr unsigned int kEuropeanSwaptionThreadsPerBlock = 256U;
inline constexpr float kEuropeanSwaptionTimeDayFraction = 1.0f / 252.0f;

enum class EuropeanSwaptionWorkDistribution {
    one_price_per_thread,
    one_price_per_block,
};

struct EuropeanSwaptionGenerationConfiguration {
    unsigned int threads_per_block;
    EuropeanSwaptionWorkDistribution work_distribution;
};

inline constexpr EuropeanSwaptionGenerationConfiguration
kDefaultEuropeanSwaptionGenerationConfiguration{
    kEuropeanSwaptionThreadsPerBlock,
    EuropeanSwaptionWorkDistribution::one_price_per_thread,
};

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
    EuropeanSwaptionGenerationConfiguration configuration =
        kDefaultEuropeanSwaptionGenerationConfiguration
) {
    if (configuration.threads_per_block == 0U) {
        throw std::invalid_argument(
            "European swaption generation requires a positive block size."
        );
    }
    constexpr PriceConstruction construction = PriceConstruction::Aligned;
    const auto& products = product_dataset.products;
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), construction
    );
    const auto block_count_for = [&](std::size_t row_count) {
        if (configuration.work_distribution
            == EuropeanSwaptionWorkDistribution::one_price_per_block) {
            return row_count;
        }
        return (row_count - 1U) / configuration.threads_per_block + 1U;
    };
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
                device_models, warmup_count,
                device_products, warmup_count,
                PriceConstruction::Aligned, warmup_count, 0U, warmup_count,
                kEuropeanSwaptionTimeDayFraction,
                configuration.threads_per_block,
                block_count_for(warmup_count),
                execution.prices()
            );
        },
        [&](auto& execution) {
            launcher(
                execution.template input<0U>(), models.size(),
                execution.template input<1U>(), products.size(),
                construction, result_count, 0U, result_count,
                kEuropeanSwaptionTimeDayFraction,
                configuration.threads_per_block,
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
            {"threads_per_block", configuration.threads_per_block},
            {"kernel_launch_count", 1U},
            {
                "work_distribution",
                configuration.work_distribution
                    == EuropeanSwaptionWorkDistribution::one_price_per_block
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
    EuropeanSwaptionGenerationConfiguration configuration =
        kDefaultEuropeanSwaptionGenerationConfiguration
) {
    if (configuration.threads_per_block == 0U) {
        throw std::invalid_argument(
            "European swaption generation requires a positive block size."
        );
    }
    constexpr PriceConstruction construction = PriceConstruction::Aligned;
    const auto& products = product_dataset.products;
    const std::size_t result_count = price_row_count(
        models.size(), curves.size(), products.size(), construction
    );
    const auto block_count_for = [&](std::size_t row_count) {
        if (configuration.work_distribution
            == EuropeanSwaptionWorkDistribution::one_price_per_block) {
            return row_count;
        }
        return (row_count - 1U) / configuration.threads_per_block + 1U;
    };
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
                execution.template input<0U>(), warmup_count,
                execution.template input<1U>(), warmup_count,
                execution.template input<2U>(), warmup_count,
                PriceConstruction::Aligned, warmup_count, 0U, warmup_count,
                kEuropeanSwaptionTimeDayFraction,
                configuration.threads_per_block,
                block_count_for(warmup_count),
                execution.prices()
            );
        },
        [&](auto& execution) {
            launcher(
                execution.template input<0U>(), models.size(),
                execution.template input<1U>(), curves.size(),
                execution.template input<2U>(), products.size(),
                construction, result_count, 0U, result_count,
                kEuropeanSwaptionTimeDayFraction,
                configuration.threads_per_block,
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
            {"threads_per_block", configuration.threads_per_block},
            {"kernel_launch_count", 1U},
            {
                "work_distribution",
                configuration.work_distribution
                    == EuropeanSwaptionWorkDistribution::one_price_per_block
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
