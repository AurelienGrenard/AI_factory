// Offline execution for static multi-asset model compositions.
#pragma once

#include "multi_asset/markovian_price_kernel.cuh"
#include "monte_carlo/pricing_moments_kernel.cuh"
#include "pricing_rows/row_index_plan.hpp"
#include "correlation/dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/pricing/equity_price_generation.cuh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

template<typename Pack>
struct MultiAssetHostModels;

template<>
struct MultiAssetHostModels<simulation::MultiAssetPack<>> {
    void validate(const std::vector<RowIndexSelector>&,
                  std::size_t) const {}
};

template<typename Head, typename... Tail>
struct MultiAssetHostModels<simulation::MultiAssetPack<Head, Tail...>> {
    using Pack = simulation::MultiAssetPack<Head, Tail...>;
    std::vector<typename Head::Parameters> models;
    MultiAssetHostModels<simulation::MultiAssetPack<Tail...>> tail;

    void validate(const std::vector<RowIndexSelector>& selectors,
                  std::size_t asset_offset) const {
        if (models.empty()
            || selectors.size() < asset_offset + Head::kAssetCount)
            throw std::invalid_argument("Invalid multi-asset model slots.");
        for (std::size_t i = 0U; i < Head::kAssetCount; ++i)
            if (selectors[asset_offset + i].source_count != models.size())
                throw std::invalid_argument(
                    "Model selector disagrees with its dataset.");
        tail.validate(selectors, asset_offset + Head::kAssetCount);
    }
};

template<typename Pack>
struct MultiAssetDeviceModels;

template<>
struct MultiAssetDeviceModels<simulation::MultiAssetPack<>> {
    explicit MultiAssetDeviceModels(
        const MultiAssetHostModels<simulation::MultiAssetPack<>>&
    ) {}
    simulation::MultiAssetPack<>::Inputs inputs(
        const RowIndexSelector* = nullptr, std::size_t = 0U
    ) const { return {}; }
};

template<typename Head, typename... Tail>
struct MultiAssetDeviceModels<simulation::MultiAssetPack<Head, Tail...>> {
    using Pack = simulation::MultiAssetPack<Head, Tail...>;
    cuda::DeviceBuffer<typename Head::Parameters> models;
    MultiAssetDeviceModels<simulation::MultiAssetPack<Tail...>> tail;

    explicit MultiAssetDeviceModels(const MultiAssetHostModels<Pack>& host)
        : models(host.models.size()), tail(host.tail) {
        models.copy_from(host.models.data());
    }

    typename Pack::Inputs inputs(
        const RowIndexSelector* selectors,
        std::size_t asset_offset = 0U
    ) const {
        return {
            {models.data(), selectors + asset_offset},
            tail.inputs(selectors, asset_offset + Head::kAssetCount)
        };
    }
};

template<typename Pack, typename Product, unsigned int TeamSize>
cuda::MonteCarloRun execute_multi_asset_prices(
    const MultiAssetHostModels<Pack>& host_models,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& expected_factors,
    const CompiledRowIndexPlan& row_plan,
    const BatchedMonteCarloProfile& profile,
    float dt,
    std::uint32_t steps_per_day
) {
    const std::size_t result_count = row_plan.result_count;
    if (row_plan.selectors.size() != Pack::kAssetCount + 2U
        || products.empty() || correlations.empty()
        || result_count == 0U || profile.paths_per_price < 2U
        || profile.results_per_launch == 0U
        || profile.block_count_limit == 0U
        || profile.blocks_per_price == 0U
        || profile.blocks_per_price > 65535U
        || profile.threads_per_block < 32U
        || profile.threads_per_block > 1024U
        || profile.threads_per_block % 32U != 0U
        || profile.threads_per_block % TeamSize != 0U
        || !(dt > 0.0f) || steps_per_day == 0U
        || expected_factors.size() != Pack::kFactorCount)
        throw std::invalid_argument("Invalid multi-asset price inputs.");
    host_models.validate(row_plan.selectors, 0U);
    if (row_plan.selectors[Pack::kAssetCount].source_count
            != products.size()
        || row_plan.selectors[Pack::kAssetCount + 1U].source_count
            != correlations.size())
        throw std::invalid_argument(
            "Product or correlation selector disagrees with its dataset.");

    constexpr std::size_t lower_count =
        Pack::kFactorCount * (Pack::kFactorCount + 1U) / 2U;
    std::vector<float> factors;
    factors.reserve(correlations.size() * lower_count);
    for (const auto& matrix : correlations) {
        if (matrix.factors != expected_factors
            || matrix.lower.size() != lower_count)
            throw std::invalid_argument(
                "Correlation factors disagree with the model slots.");
        factors.insert(factors.end(), matrix.lower.begin(), matrix.lower.end());
    }
    for (const auto& product : products) {
        if constexpr (requires { Product::valid_parameters(
            product, steps_per_day); }) {
            if (!Product::valid_parameters(product, steps_per_day))
                throw std::invalid_argument(
                    "Invalid multi-asset product parameters.");
        }
        const std::uint64_t steps =
            static_cast<std::uint64_t>(product.maturity_days)
                * steps_per_day;
        if (steps == 0U || steps > philox::kMaximumDomainStep)
            throw std::invalid_argument(
                "Multi-asset maturity exceeds the Philox step domain.");
    }
    const auto maximum_launch_rows =
        std::min(result_count, profile.results_per_launch);
    if (maximum_launch_rows > std::numeric_limits<std::size_t>::max()
        / profile.blocks_per_price / sizeof(reductions::MomentSums))
        throw std::overflow_error("Multi-asset moment workspace overflow.");

    const auto wall_start = std::chrono::steady_clock::now();
    MultiAssetDeviceModels<Pack> device_models(host_models);
    cuda::DeviceBuffer<RowIndexSelector> device_selectors(
        row_plan.selectors.size());
    device_selectors.copy_from(row_plan.selectors.data());
    const auto launch_batch = [&](auto& execution, std::size_t offset,
                                  std::size_t count, std::size_t paths) {
        const std::size_t blocks = std::min(
            profile.blocks_per_price,
            1U + (paths - 1U) / profile.threads_per_block
        );
        auto* partial = blocks > 1U
            ? reinterpret_cast<reductions::MomentSums*>(execution.workspace())
            : nullptr;
        monte_carlo::launch_multi_asset_cuda<Pack, Product, TeamSize>(
            {device_models.inputs(device_selectors.data()),
             execution.template input<0U>(),
             execution.template input<1U>(),
             device_selectors.data()},
            result_count, offset, count, paths, dt, steps_per_day,
            profile.threads_per_block,
            std::min(count, profile.block_count_limit), profile.seed,
            execution.prices(), execution.standard_errors(), blocks, partial
        );
        if (blocks > 1U) {
            monte_carlo::finalize_price_batch_kernel
                <<<static_cast<unsigned int>(count), 128U,
                    8U * sizeof(double)>>>(
                    partial, blocks, paths, offset,
                    execution.prices(), execution.standard_errors()
                );
            check_cuda(cudaGetLastError(), "multi-asset batch finalize");
        }
    };
    auto run = cuda::run_monte_carlo_with_workspace(
        cuda::inputs(products, factors), result_count,
        profile.blocks_per_price > 1U
            ? maximum_launch_rows * profile.blocks_per_price
                * sizeof(reductions::MomentSums)
            : 0U,
        [&](auto& execution) {
            launch_batch(
                execution, 0U,
                std::min<std::size_t>(result_count,
                    std::min<std::size_t>(profile.results_per_launch, 8U)),
                std::min<std::size_t>(profile.paths_per_price, 256U)
            );
        },
        [&](auto& execution) {
            for (std::size_t offset = 0U; offset < result_count;
                 offset += profile.results_per_launch) {
                const auto count = std::min(
                    profile.results_per_launch, result_count - offset
                );
                launch_batch(execution, offset, count,
                             profile.paths_per_price);
            }
        }
    );
    run.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();
    return run;
}

}  // namespace ai_factory::workbench::offline::pricing
