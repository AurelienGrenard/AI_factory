// Offline mixed Markovian/rough price execution.
#pragma once

#include "common/monte_carlo/multi_asset_rough_kernel.cuh"
#include "common/volterra/hybrid_fft_volatility.cuh"
#include "tools/pricing/multi_asset_price_generation.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

template<typename MarkovPack, typename RoughPolicy, typename Product,
         std::size_t RoughCount = 1U,unsigned TeamSize=1U>
cuda::MonteCarloRun execute_mixed_rough_prices(
    const MultiAssetHostModels<MarkovPack>& markov_models,
    const std::vector<typename RoughPolicy::Parameters>& rough_models,
    const std::vector<std::uint32_t>& rough_scenario_indices,
    std::size_t scenario_count,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& external_factor_order,
    const std::vector<std::string>& internal_factor_order,
    const std::vector<monte_carlo::MultiAssetPriceRow>& rows,
    std::size_t paths_per_price,
    std::size_t path_chunk_size,
    float dt,
    std::uint32_t steps_per_day,
    std::uint32_t direct_tile_size,
    std::uint64_t base_seed
) {
    static_assert(RoughCount > 0U);
    constexpr bool causal_rough = requires {
        typename RoughPolicy::VolatilityState;
    };
    constexpr std::size_t dimension = MarkovPack::kFactorCount
        + RoughCount * RoughPolicy::kBrownianDimension;
    constexpr std::size_t lower_count = dimension * (dimension + 1U) / 2U;
    constexpr std::uint32_t auxiliary_source = static_cast<std::uint32_t>(
        2U + MarkovPack::kAssetCount
    );
    static_assert(auxiliary_source + RoughCount <= 256U);
    if (scenario_count == 0U || rough_models.empty()
        || rough_scenario_indices.size() != scenario_count * RoughCount
        || products.empty() || correlations.empty() || rows.empty()
        || paths_per_price < 128U || paths_per_price % 128U != 0U
        || path_chunk_size == 0U || path_chunk_size % 128U != 0U
        || !(dt > 0.0f) || steps_per_day == 0U
        || external_factor_order.size() != dimension
        || internal_factor_order.size() != dimension
        || (direct_tile_size != 32U && direct_tile_size != 64U)) {
        throw std::invalid_argument("Invalid mixed rough price inputs.");
    }
    std::vector<std::string> required_internal;
    required_internal.reserve(dimension);
    if constexpr (RoughPolicy::kBrownianDimension == 2U) {
        for (std::size_t asset = 0U; asset < RoughCount; ++asset) {
            required_internal.push_back(external_factor_order[
                MarkovPack::kFactorCount + 2U * asset + 1U
            ]);
        }
    } else {
        for (std::size_t asset = 0U; asset < RoughCount; ++asset) {
            required_internal.push_back(external_factor_order[
                MarkovPack::kFactorCount + asset
            ]);
        }
    }
    for (std::size_t factor = 0U; factor < MarkovPack::kFactorCount;
         ++factor) {
        required_internal.push_back(external_factor_order[factor]);
    }
    if constexpr (RoughPolicy::kBrownianDimension == 2U) {
        for (std::size_t asset = 0U; asset < RoughCount; ++asset) {
            required_internal.push_back(external_factor_order[
                MarkovPack::kFactorCount + 2U * asset
            ]);
        }
    }
    if (internal_factor_order != required_internal) {
        throw std::invalid_argument(
            "Internal rough factor order does not match model slots."
        );
    }
    markov_models.validate(scenario_count);
    for (const auto index : rough_scenario_indices) {
        if (index >= rough_models.size())
            throw std::invalid_argument("Rough model index exceeds its dataset.");
    }
    std::vector<float> packed_factors;
    packed_factors.reserve(correlations.size() * lower_count);
    std::vector<correlation::MatrixRow> ordered;
    ordered.reserve(correlations.size());
    for (const auto& matrix : correlations) {
        if (matrix.factors != external_factor_order)
            throw std::invalid_argument("External correlation factors disagree with model slots.");
        ordered.push_back(correlation::reorder_factors(
            matrix, internal_factor_order
        ));
        packed_factors.insert(packed_factors.end(),
            ordered.back().lower.begin(), ordered.back().lower.end());
    }
    std::uint32_t max_steps = 0U;
    for (const auto& product : products) {
        const auto count = static_cast<std::uint64_t>(product.maturity_days)
            * steps_per_day;
        if (count == 0U || count > (causal_rough ? 8192U : 4096U))
            throw std::invalid_argument("Mixed rough maturity exceeds FFT capacity.");
        max_steps = std::max(max_steps, static_cast<std::uint32_t>(count));
        if constexpr (requires { Product::valid_parameters(
            product, steps_per_day); }) {
            if (!Product::valid_parameters(product, steps_per_day))
                throw std::invalid_argument(
                    "Invalid mixed rough product parameters.");
        }
    }
    for (const auto& row : rows) {
        if (row.scenario_index >= scenario_count
            || row.product_index >= products.size()
            || row.correlation_index >= ordered.size()
            || !std::isfinite(row.discount_factor)
            || !(row.discount_factor > 0.0f)) {
            throw std::invalid_argument("Invalid mixed rough price row.");
        }
        const auto& matrix = ordered[row.correlation_index];
        markov_models.template validate_internal_correlation<0U, RoughCount>(
            row.scenario_index, matrix
        );
        if constexpr (RoughPolicy::kBrownianDimension == 2U) {
            const auto& lower = matrix.lower;
            for (std::size_t asset = 0U; asset < RoughCount; ++asset) {
                const float rho = rough_models[
                    rough_scenario_indices[
                        static_cast<std::size_t>(row.scenario_index)
                            * RoughCount + asset
                    ]
                ].rho;
                const std::size_t spot = RoughCount
                    + MarkovPack::kFactorCount + asset;
                double actual = 0.0;
                for (std::size_t k = 0U; k <= asset; ++k) {
                    actual += static_cast<double>(
                        lower[asset * (asset + 1U) / 2U + k]
                    ) * static_cast<double>(
                        lower[spot * (spot + 1U) / 2U + k]
                    );
                }
                if (!std::isfinite(rho) || std::abs(actual - rho) > 2.0e-5)
                    throw std::invalid_argument(
                        "Rough spot/variance rho disagrees with correlation matrix."
                    );
            }
        }
    }
    const std::size_t chunk = std::min(paths_per_price, path_chunk_size);
    const std::size_t volatility_workspace_bytes = [&] {
        if constexpr (causal_rough)
            return volterra::causal_fft::plan_volatility_workspace<
                RoughPolicy>(max_steps, chunk).bytes;
        else
            return volterra::hybrid_fft::plan_correlated_volatility_workspace<
                RoughPolicy>(max_steps, chunk).bytes;
    }();
    const auto wall_start = std::chrono::steady_clock::now();
    MultiAssetDeviceModels<MarkovPack> device_markov(markov_models);
    cuda::DeviceBuffer<typename RoughPolicy::Parameters> device_rough(
        rough_models.size());
    cuda::DeviceBuffer<std::uint32_t> device_rough_indices(
        rough_scenario_indices.size());
    cuda::DeviceBuffer<typename Product::ProductParameters> device_products(
        products.size());
    cuda::DeviceBuffer<monte_carlo::MultiAssetPriceRow> device_rows(rows.size());
    cuda::DeviceBuffer<float> device_factors(packed_factors.size());
    cuda::DeviceBuffer<float> device_prices(rows.size());
    cuda::DeviceBuffer<float> device_errors(rows.size());
    cuda::DeviceBuffer<volterra::PartialMoments> device_moments(
        paths_per_price / 128U);
    cuda::DeviceBuffer<float> device_volatility(
        RoughCount * chunk * max_steps);
    cuda::DeviceBuffer<std::uint8_t> device_workspace(
        volatility_workspace_bytes);
    device_rough.copy_from(rough_models.data());
    device_rough_indices.copy_from(rough_scenario_indices.data());
    device_products.copy_from(products.data());
    device_rows.copy_from(rows.data());
    device_factors.copy_from(packed_factors.data());
    const monte_carlo::MixedAssetDeviceInputs<
        MarkovPack, RoughPolicy, Product, RoughCount
    > inputs{
        device_markov.inputs(), device_rough.data(),
        device_rough_indices.data(), device_products.data(),
        device_rows.data(), device_factors.data()
    };
    cuda::Event start, stop;
    check_cuda(cudaEventRecord(start.get()), "mixed rough timer start");
    for (std::size_t result = 0U; result < rows.size(); ++result) {
        const auto steps = products[rows[result].product_index].maturity_days
            * steps_per_day;
        for (std::size_t offset = 0U; offset < paths_per_price;
             offset += chunk) {
            const auto count = std::min(chunk, paths_per_price - offset);
            for (std::size_t asset = 0U; asset < RoughCount; ++asset) {
                if constexpr (causal_rough) {
                    volterra::causal_fft::launch_volatility_factor_cuda<
                        RoughPolicy, dimension
                    >(
                        device_rough.data(), device_rough_indices.data(),
                        device_rows.data(), device_factors.data(), result,
                        offset, count, dt, steps, direct_tile_size,
                        base_seed, device_workspace.data(),
                        device_workspace.size(),
                        device_volatility.data() + asset * count * steps,
                        asset, auxiliary_source
                            + static_cast<std::uint32_t>(asset),
                        RoughCount, asset
                    );
                } else {
                    volterra::hybrid_fft::launch_correlated_hybrid_volatility_cuda<
                        RoughPolicy, dimension
                    >(
                        device_rough.data(), device_rough_indices.data(),
                        device_rows.data(), device_factors.data(), result,
                        offset, count, dt, steps, base_seed,
                        device_workspace.data(), device_workspace.size(),
                        device_volatility.data() + asset * count * steps,
                        asset, auxiliary_source
                            + static_cast<std::uint32_t>(asset),
                        RoughCount, asset
                    );
                }
            }
            monte_carlo::launch_joint_rough_cuda<
                MarkovPack, RoughPolicy, Product, false, RoughCount,
                TeamSize
            >(
                inputs, result, offset, count, steps, steps_per_day,
                dt, base_seed, device_volatility.data(),
                device_moments.data()
            );
        }
        volterra::finalize_price_kernel<<<1U, 128U,
            2U * (128U / 32U) * sizeof(double)>>>(
                device_moments.data(), paths_per_price / 128U,
                paths_per_price, result, device_prices.data(),
                device_errors.data()
            );
        check_cuda(cudaGetLastError(), "mixed rough price finalize");
    }
    check_cuda(cudaEventRecord(stop.get()), "mixed rough timer stop");
    check_cuda(cudaEventSynchronize(stop.get()), "mixed rough timer wait");
    float milliseconds = 0.0f;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
        "mixed rough elapsed time");
    cuda::MonteCarloRun run;
    run.prices.resize(rows.size());
    run.standard_errors.resize(rows.size());
    device_prices.copy_to(run.prices.data());
    device_errors.copy_to(run.standard_errors.data());
    run.kernel_seconds = static_cast<double>(milliseconds) * 1.0e-3;
    run.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();
    return run;
}

}  // namespace ai_factory::workbench::offline::pricing
