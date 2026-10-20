// Offline execution for static multi-asset model compositions.
#pragma once

#include "common/monte_carlo/multi_asset_kernel.cuh"
#include "correlation/dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/pricing/equity_price_generation.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

template<typename Pack>
struct MultiAssetHostModels;

template<>
struct MultiAssetHostModels<simulation::MultiAssetPack<>> {
    void validate(std::size_t) const {}
    template<std::size_t, std::size_t>
    void validate_internal_correlation(
        std::uint32_t, const correlation::MatrixRow&
    ) const {}
};

template<typename Head, typename... Tail>
struct MultiAssetHostModels<simulation::MultiAssetPack<Head, Tail...>> {
    using Pack = simulation::MultiAssetPack<Head, Tail...>;
    std::vector<typename Head::Parameters> models;
    std::vector<std::uint32_t> scenario_indices;
    MultiAssetHostModels<simulation::MultiAssetPack<Tail...>> tail;

    void validate(std::size_t scenarios) const {
        if (models.empty()
            || scenario_indices.size() != scenarios * Head::kAssetCount) {
            throw std::invalid_argument(
                "Multi-asset model group has invalid scenario indices."
            );
        }
        for (const auto index : scenario_indices) {
            if (index >= models.size()) {
                throw std::invalid_argument(
                    "Multi-asset model index exceeds its dataset."
                );
            }
        }
        tail.validate(scenarios);
    }

    template<std::size_t AssetOffset = 0U, std::size_t FactorOffset = 0U>
    void validate_internal_correlation(
        std::uint32_t scenario, const correlation::MatrixRow& matrix
    ) const {
        if constexpr (Head::Dynamics::kBrownianDimension == 2U
            && (requires(const typename Head::Parameters& parameters) {
                    parameters.rho;
                } || requires(const typename Head::Parameters& parameters) {
                    parameters.correlation;
                })) {
            const auto& lower = matrix.lower;
            for (std::size_t i = 0U; i < Head::kAssetCount; ++i) {
                const auto model_index = scenario_indices[
                    scenario * Head::kAssetCount + i
                ];
                const float rho = [&] {
                    if constexpr (requires(const typename Head::Parameters& p) {
                        p.rho;
                    }) {
                        return models[model_index].rho;
                    } else {
                        return models[model_index].correlation;
                    }
                }();
                const std::size_t first = FactorOffset + 2U * i;
                const std::size_t second = first + 1U;
                double correlation = 0.0;
                for (std::size_t k = 0U; k <= first; ++k) {
                    correlation += static_cast<double>(
                        lower[first * (first + 1U) / 2U + k]
                    ) * static_cast<double>(
                        lower[second * (second + 1U) / 2U + k]
                    );
                }
                if (!std::isfinite(rho)
                    || std::abs(correlation - rho) > 2.0e-5) {
                    throw std::invalid_argument(
                        "Model rho disagrees with its two Brownian factors."
                    );
                }
            }
        }
        if constexpr (sizeof...(Tail) > 0U) {
            tail.template validate_internal_correlation<
                AssetOffset + Head::kAssetCount,
                FactorOffset + Head::kFactorCount
            >(scenario, matrix);
        }
    }
};

template<typename Pack>
struct MultiAssetDeviceModels;

template<>
struct MultiAssetDeviceModels<simulation::MultiAssetPack<>> {
    explicit MultiAssetDeviceModels(
        const MultiAssetHostModels<simulation::MultiAssetPack<>>&
    ) {}
    simulation::MultiAssetPack<>::Inputs inputs() const { return {}; }
};

template<typename Head, typename... Tail>
struct MultiAssetDeviceModels<simulation::MultiAssetPack<Head, Tail...>> {
    using Pack = simulation::MultiAssetPack<Head, Tail...>;
    cuda::DeviceBuffer<typename Head::Parameters> models;
    cuda::DeviceBuffer<std::uint32_t> indices;
    MultiAssetDeviceModels<simulation::MultiAssetPack<Tail...>> tail;

    explicit MultiAssetDeviceModels(
        const MultiAssetHostModels<Pack>& host
    ) : models(host.models.size()),
        indices(host.scenario_indices.size()),
        tail(host.tail) {
        models.copy_from(host.models.data());
        indices.copy_from(host.scenario_indices.data());
    }

    typename Pack::Inputs inputs() const {
        return {
            {models.data(), indices.data()},
            tail.inputs()
        };
    }
};

template<typename Pack, typename Product, unsigned int TeamSize>
cuda::MonteCarloRun execute_multi_asset_prices(
    const MultiAssetHostModels<Pack>& host_models,
    std::size_t scenario_count,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& expected_factors,
    const std::vector<monte_carlo::MultiAssetPriceRow>& rows,
    const BatchedMonteCarloProfile& profile,
    float dt,
    std::uint32_t steps_per_day
) {
    if (scenario_count == 0U || products.empty() || correlations.empty()
        || rows.empty() || profile.paths_per_price < 2U
        || profile.results_per_launch == 0U
        || profile.block_count_limit == 0U
        || !(dt > 0.0f) || steps_per_day == 0U
        || expected_factors.size() != Pack::kFactorCount) {
        throw std::invalid_argument("Invalid multi-asset price inputs.");
    }
    host_models.validate(scenario_count);
    constexpr std::size_t lower_count =
        Pack::kFactorCount * (Pack::kFactorCount + 1U) / 2U;
    std::vector<float> factors;
    factors.reserve(correlations.size() * lower_count);
    for (const auto& matrix : correlations) {
        if (matrix.factors != expected_factors
            || matrix.lower.size() != lower_count) {
            throw std::invalid_argument(
                "Correlation factors disagree with the model slots."
            );
        }
        factors.insert(
            factors.end(), matrix.lower.begin(), matrix.lower.end()
        );
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
        if (steps == 0U || steps > philox::kMaximumDomainStep) {
            throw std::invalid_argument(
                "Multi-asset maturity exceeds the Philox step domain."
            );
        }
    }
    for (const auto& row : rows) {
        if (row.scenario_index >= scenario_count
            || row.product_index >= products.size()
            || row.correlation_index >= correlations.size()
            || !std::isfinite(row.discount_factor)
            || !(row.discount_factor > 0.0f)) {
            throw std::invalid_argument("Invalid multi-asset price row.");
        }
        host_models.template validate_internal_correlation<0U, 0U>(
            row.scenario_index, correlations[row.correlation_index]
        );
    }

    const auto wall_start = std::chrono::steady_clock::now();
    MultiAssetDeviceModels<Pack> device_models(host_models);
    auto run = cuda::run_monte_carlo(
        cuda::inputs(products, rows, factors),
        rows.size(),
        [&](auto& execution) {
            const auto count = std::min<std::size_t>(
                rows.size(), std::min<std::size_t>(
                    profile.results_per_launch, 8U
                )
            );
            monte_carlo::launch_multi_asset_cuda<
                Pack, Product, TeamSize
            >(
                {device_models.inputs(),
                 execution.template input<0U>(),
                 execution.template input<1U>(),
                 execution.template input<2U>()},
                rows.size(), 0U, count,
                std::min<std::size_t>(profile.paths_per_price, 256U),
                dt, steps_per_day, profile.threads_per_block,
                std::min(count, profile.block_count_limit), profile.seed,
                execution.prices(), execution.standard_errors()
            );
        },
        [&](auto& execution) {
            for (std::size_t offset = 0U; offset < rows.size();
                 offset += profile.results_per_launch) {
                const auto count = std::min(
                    profile.results_per_launch, rows.size() - offset
                );
                monte_carlo::launch_multi_asset_cuda<
                    Pack, Product, TeamSize
                >(
                    {device_models.inputs(),
                     execution.template input<0U>(),
                     execution.template input<1U>(),
                     execution.template input<2U>()},
                    rows.size(), offset, count,
                    profile.paths_per_price, dt, steps_per_day,
                    profile.threads_per_block,
                    std::min(count, profile.block_count_limit), profile.seed,
                    execution.prices(), execution.standard_errors()
                );
            }
        }
    );
    run.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();
    return run;
}

}  // namespace ai_factory::workbench::offline::pricing
