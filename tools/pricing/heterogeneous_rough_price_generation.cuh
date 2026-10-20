// Static rough-type composition for volatility-first multi-asset pricing.
#pragma once

#include "common/monte_carlo/multi_asset_heterogeneous_rough_kernel.cuh"
#include "common/volterra/causal_fft_volatility.cuh"
#include "common/volterra/hybrid_fft_volatility.cuh"
#include "tools/pricing/multi_asset_price_generation.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

template<typename Pack>
struct MultiRoughHostModels;

template<>
struct MultiRoughHostModels<simulation::MultiRoughPack<>> {
    void validate(std::size_t) const {}
};

template<typename Head,typename... Tail>
struct MultiRoughHostModels<simulation::MultiRoughPack<Head,Tail...>> {
    using Rest=simulation::MultiRoughPack<Tail...>;
    std::vector<typename Head::Parameters> models;
    std::vector<std::uint32_t> scenario_indices;
    MultiRoughHostModels<Rest> tail;

    void validate(std::size_t scenarios) const {
        if(models.empty() || scenario_indices.size()!=scenarios)
            throw std::invalid_argument("invalid rough model scenario indices");
        for(const auto index:scenario_indices)
            if(index>=models.size())
                throw std::invalid_argument("rough model index exceeds dataset");
        tail.validate(scenarios);
    }

    template<std::size_t RoughCount,std::size_t MarkovFactors,
             std::size_t AssetOffset=0U,std::size_t SpotOffset=0U>
    void validate_internal_correlation(
        std::uint32_t scenario,const correlation::MatrixRow& matrix
    ) const {
        if constexpr (Head::kBrownianDimension==2U) {
            const float rho=models[scenario_indices[scenario]].rho;
            const std::size_t spot=RoughCount+MarkovFactors+SpotOffset;
            double actual=0.0;
            for(std::size_t k=0U;k<=AssetOffset;++k)
                actual+=static_cast<double>(matrix.lower[
                    AssetOffset*(AssetOffset+1U)/2U+k])
                    *static_cast<double>(matrix.lower[
                        spot*(spot+1U)/2U+k]);
            if(!std::isfinite(rho) || std::abs(actual-rho)>2.0e-5)
                throw std::invalid_argument(
                    "rough spot/vol rho disagrees with matrix");
        }
        if constexpr (Rest::kAssetCount>0U)
            tail.template validate_internal_correlation<
                RoughCount,MarkovFactors,AssetOffset+1U,
                SpotOffset+(Head::kBrownianDimension==2U ? 1U : 0U)
            >(scenario,matrix);
    }
};

template<typename Policy, std::size_t Count>
struct MultiRoughHostModels<
    simulation::HomogeneousRoughPack<Policy, Count>> {
    std::vector<typename Policy::Parameters> models;
    std::vector<std::uint32_t> scenario_indices;

    void validate(std::size_t scenarios) const {
        if (models.empty() || scenario_indices.size() != scenarios * Count)
            throw std::invalid_argument("invalid homogeneous rough indices");
        for (const auto index : scenario_indices)
            if (index >= models.size())
                throw std::invalid_argument(
                    "rough model index exceeds dataset");
    }

    template<std::size_t RoughCount, std::size_t MarkovFactors>
    void validate_internal_correlation(
        std::uint32_t scenario, const correlation::MatrixRow& matrix
    ) const {
        static_assert(RoughCount == Count);
        if constexpr (Policy::kBrownianDimension == 2U) {
            for (std::size_t asset = 0U; asset < Count; ++asset) {
                const float rho = models[
                    scenario_indices[scenario * Count + asset]].rho;
                const std::size_t spot =
                    Count + MarkovFactors + asset;
                double actual = 0.0;
                for (std::size_t k = 0U; k <= asset; ++k)
                    actual += static_cast<double>(matrix.lower[
                        asset * (asset + 1U) / 2U + k])
                        * static_cast<double>(matrix.lower[
                            spot * (spot + 1U) / 2U + k]);
                if (!std::isfinite(rho)
                    || std::abs(actual - rho) > 2.0e-5)
                    throw std::invalid_argument(
                        "rough spot/vol rho disagrees with matrix");
            }
        }
    }
};

template<typename Pack>
struct MultiRoughDeviceModels;

template<>
struct MultiRoughDeviceModels<simulation::MultiRoughPack<>> {
    explicit MultiRoughDeviceModels(
        const MultiRoughHostModels<simulation::MultiRoughPack<>>&
    ) {}
    simulation::MultiRoughPack<>::Inputs inputs() const { return {}; }
    static std::size_t workspace_bytes(std::uint32_t,std::size_t) {
        return 0U;
    }
};

template<typename Head,typename... Tail>
struct MultiRoughDeviceModels<simulation::MultiRoughPack<Head,Tail...>> {
    using Pack=simulation::MultiRoughPack<Head,Tail...>;
    using Rest=simulation::MultiRoughPack<Tail...>;
    cuda::DeviceBuffer<typename Head::Parameters> models;
    cuda::DeviceBuffer<std::uint32_t> indices;
    MultiRoughDeviceModels<Rest> tail;

    explicit MultiRoughDeviceModels(const MultiRoughHostModels<Pack>& host)
        :models(host.models.size()),indices(host.scenario_indices.size()),
         tail(host.tail) {
        models.copy_from(host.models.data());
        indices.copy_from(host.scenario_indices.data());
    }
    typename Pack::Inputs inputs() const {
        return {models.data(),indices.data(),tail.inputs()};
    }
    static std::size_t workspace_bytes(
        std::uint32_t steps,std::size_t chunk
    ) {
        const std::size_t head_bytes=[](std::uint32_t count,
                                        std::size_t paths) {
            if constexpr (requires { typename Head::VolatilityState; })
                return volterra::causal_fft::plan_volatility_workspace<
                    Head>(count,paths).bytes;
            else return volterra::hybrid_fft::
                plan_correlated_volatility_workspace<Head>(
                    count,paths).bytes;
        }(steps,chunk);
        return std::max(head_bytes,
            MultiRoughDeviceModels<Rest>::workspace_bytes(steps,chunk));
    }

    template<std::size_t Dimension,std::size_t MarkovAssets,
             std::size_t AssetOffset=0U>
    void launch_volatilities(
        const monte_carlo::MultiAssetPriceRow* price_rows,
        const float* packed_factors,
        std::size_t result_index,std::size_t path_offset,
        std::size_t path_count,float dt,std::uint32_t steps,
        std::uint32_t direct_tile_size,std::uint64_t seed,
        void* workspace,std::size_t workspace_bytes,
        float* volatility_paths
    ) const {
        constexpr std::uint32_t source=static_cast<std::uint32_t>(
            2U+MarkovAssets+AssetOffset);
        float* output=volatility_paths+AssetOffset*path_count*steps;
        if constexpr (requires { typename Head::VolatilityState; }) {
            volterra::causal_fft::launch_volatility_factor_cuda<
                Head,Dimension>(
                    models.data(),indices.data(),price_rows,packed_factors,
                    result_index,path_offset,path_count,dt,steps,
                    direct_tile_size,seed,workspace,workspace_bytes,
                    output,AssetOffset,source);
        } else {
            volterra::hybrid_fft::launch_correlated_hybrid_volatility_cuda<
                Head,Dimension>(
                    models.data(),indices.data(),price_rows,packed_factors,
                    result_index,path_offset,path_count,dt,steps,seed,
                    workspace,workspace_bytes,output,AssetOffset,source);
        }
        if constexpr (Rest::kAssetCount>0U)
            tail.template launch_volatilities<
                Dimension,MarkovAssets,AssetOffset+1U>(
                    price_rows,packed_factors,result_index,path_offset,
                    path_count,dt,steps,direct_tile_size,seed,
                    workspace,workspace_bytes,volatility_paths);
    }
};

template<typename Policy, std::size_t Count>
struct MultiRoughDeviceModels<
    simulation::HomogeneousRoughPack<Policy, Count>> {
    using Pack = simulation::HomogeneousRoughPack<Policy, Count>;
    cuda::DeviceBuffer<typename Policy::Parameters> models;
    cuda::DeviceBuffer<std::uint32_t> indices;

    explicit MultiRoughDeviceModels(
        const MultiRoughHostModels<Pack>& host
    ) : models(host.models.size()), indices(host.scenario_indices.size()) {
        models.copy_from(host.models.data());
        indices.copy_from(host.scenario_indices.data());
    }

    typename Pack::Inputs inputs() const {
        return {models.data(), indices.data()};
    }

    static std::size_t workspace_bytes(
        std::uint32_t steps, std::size_t chunk
    ) {
        if constexpr (Pack::kAllCausal)
            return volterra::causal_fft::plan_volatility_workspace<
                Policy>(steps, chunk).bytes;
        else
            return volterra::hybrid_fft::
                plan_correlated_volatility_workspace<Policy>(
                    steps, chunk).bytes;
    }

    template<std::size_t Dimension, std::size_t MarkovAssets>
    void launch_volatilities(
        const monte_carlo::MultiAssetPriceRow* price_rows,
        const float* packed_factors,
        std::size_t result_index, std::size_t path_offset,
        std::size_t path_count, float dt, std::uint32_t steps,
        std::uint32_t direct_tile_size, std::uint64_t seed,
        void* workspace, std::size_t workspace_bytes,
        float* volatility_paths
    ) const {
        for (std::size_t asset = 0U; asset < Count; ++asset) {
            float* output = volatility_paths
                + asset * path_count * steps;
            const auto source = static_cast<std::uint32_t>(
                2U + MarkovAssets + asset);
            if constexpr (Pack::kAllCausal) {
                volterra::causal_fft::launch_volatility_factor_cuda<
                    Policy, Dimension>(
                        models.data(), indices.data(), price_rows,
                        packed_factors, result_index, path_offset,
                        path_count, dt, steps, direct_tile_size, seed,
                        workspace, workspace_bytes, output,
                        asset, source, Count, asset);
            } else {
                volterra::hybrid_fft::
                    launch_correlated_hybrid_volatility_cuda<
                        Policy, Dimension>(
                            models.data(), indices.data(), price_rows,
                            packed_factors, result_index, path_offset,
                            path_count, dt, steps, seed, workspace,
                            workspace_bytes, output, asset, source,
                            Count, asset);
            }
        }
    }
};

template<typename MarkovPack,typename RoughPack,typename Product,
         unsigned TeamSize=1U>
cuda::MonteCarloRun execute_heterogeneous_rough_prices(
    const MultiAssetHostModels<MarkovPack>& markov_models,
    const MultiRoughHostModels<RoughPack>& rough_models,
    std::size_t scenario_count,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& external_factor_order,
    const std::vector<std::string>& internal_factor_order,
    const std::vector<monte_carlo::MultiAssetPriceRow>& rows,
    std::size_t paths_per_price,std::size_t path_chunk_size,
    float dt,std::uint32_t steps_per_day,std::uint32_t direct_tile_size,
    std::uint64_t base_seed
) {
    static_assert(RoughPack::kAssetCount>0U);
    static_assert(MarkovPack::kAssetCount+RoughPack::kAssetCount<=254U);
    constexpr std::size_t dimension=RoughPack::kAssetCount
        +MarkovPack::kFactorCount+RoughPack::kTwoFactorCount;
    constexpr std::size_t lower_count=dimension*(dimension+1U)/2U;
    if(scenario_count==0U || products.empty() || correlations.empty()
        || rows.empty() || paths_per_price<128U
        || paths_per_price%128U!=0U || path_chunk_size==0U
        || path_chunk_size%128U!=0U || !(dt>0.0f)
        || steps_per_day==0U
        || (direct_tile_size!=32U && direct_tile_size!=64U)
        || external_factor_order.size()!=dimension
        || internal_factor_order.size()!=dimension)
        throw std::invalid_argument("invalid heterogeneous rough inputs");
    markov_models.validate(scenario_count);
    rough_models.validate(scenario_count);
    std::vector<std::string> variance_names,spot_names;
    RoughPack::append_factor_names(external_factor_order,
        MarkovPack::kFactorCount,variance_names,spot_names);
    std::vector<std::string> required=variance_names;
    required.insert(required.end(),external_factor_order.begin(),
        external_factor_order.begin()+MarkovPack::kFactorCount);
    required.insert(required.end(),spot_names.begin(),spot_names.end());
    if(internal_factor_order!=required)
        throw std::invalid_argument("heterogeneous rough factor order mismatch");
    std::vector<correlation::MatrixRow> ordered;
    std::vector<float> factors;
    ordered.reserve(correlations.size());
    factors.reserve(correlations.size()*lower_count);
    for(const auto& matrix:correlations) {
        if(matrix.factors!=external_factor_order)
            throw std::invalid_argument("heterogeneous external factors mismatch");
        ordered.push_back(correlation::reorder_factors(
            matrix,internal_factor_order));
        factors.insert(factors.end(),ordered.back().lower.begin(),
                       ordered.back().lower.end());
    }
    std::uint32_t max_steps=0U;
    for(const auto& product:products) {
        if constexpr (requires { Product::valid_parameters(
            product,steps_per_day); }) {
            if(!Product::valid_parameters(product,steps_per_day))
                throw std::invalid_argument("invalid heterogeneous product");
        }
        const auto count=static_cast<std::uint64_t>(
            product.maturity_days)*steps_per_day;
        const auto limit=RoughPack::kAllCausal ? 8192U : 4096U;
        if(count==0U || count>limit)
            throw std::invalid_argument("heterogeneous rough FFT capacity exceeded");
        max_steps=std::max(max_steps,static_cast<std::uint32_t>(count));
    }
    for(const auto& row:rows) {
        if(row.scenario_index>=scenario_count
            || row.product_index>=products.size()
            || row.correlation_index>=ordered.size()
            || !std::isfinite(row.discount_factor)
            || !(row.discount_factor>0.0f))
            throw std::invalid_argument("invalid heterogeneous price row");
        const auto& matrix=ordered[row.correlation_index];
        markov_models.template validate_internal_correlation<
            0U,RoughPack::kAssetCount>(row.scenario_index,matrix);
        rough_models.template validate_internal_correlation<
            RoughPack::kAssetCount,MarkovPack::kFactorCount>(
                row.scenario_index,matrix);
    }
    const auto chunk=std::min(paths_per_price,path_chunk_size);
    const auto workspace_bytes=MultiRoughDeviceModels<RoughPack>::
        workspace_bytes(max_steps,chunk);
    const auto wall_start=std::chrono::steady_clock::now();
    MultiAssetDeviceModels<MarkovPack> device_markov(markov_models);
    MultiRoughDeviceModels<RoughPack> device_rough(rough_models);
    cuda::DeviceBuffer<typename Product::ProductParameters>
        device_products(products.size());
    cuda::DeviceBuffer<monte_carlo::MultiAssetPriceRow> device_rows(rows.size());
    cuda::DeviceBuffer<float> device_factors(factors.size());
    cuda::DeviceBuffer<float> device_prices(rows.size());
    cuda::DeviceBuffer<float> device_errors(rows.size());
    cuda::DeviceBuffer<volterra::PartialMoments>
        device_moments(paths_per_price/128U);
    cuda::DeviceBuffer<float> device_volatility(
        RoughPack::kAssetCount*chunk*max_steps);
    cuda::DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
    device_products.copy_from(products.data());
    device_rows.copy_from(rows.data());
    device_factors.copy_from(factors.data());
    const monte_carlo::HeterogeneousRoughDeviceInputs<
        MarkovPack,RoughPack,Product> inputs{
            device_markov.inputs(),device_rough.inputs(),
            device_products.data(),device_rows.data(),device_factors.data()};
    cuda::Event start,stop;
    check_cuda(cudaEventRecord(start.get()),"heterogeneous timer start");
    for(std::size_t result=0U;result<rows.size();++result) {
        const auto steps=products[rows[result].product_index].maturity_days
            *steps_per_day;
        for(std::size_t offset=0U;offset<paths_per_price;offset+=chunk) {
            const auto count=std::min(chunk,paths_per_price-offset);
            device_rough.template launch_volatilities<
                dimension,MarkovPack::kAssetCount>(
                    device_rows.data(),device_factors.data(),result,
                    offset,count,dt,steps,direct_tile_size,base_seed,
                    workspace.data(),workspace.size(),
                    device_volatility.data());
            monte_carlo::launch_heterogeneous_rough_cuda<
                MarkovPack,RoughPack,Product,false,TeamSize>(
                    inputs,result,offset,count,steps,steps_per_day,dt,
                    base_seed,device_volatility.data(),
                    device_moments.data());
        }
        volterra::finalize_price_kernel<<<1U,128U,
            2U*(128U/32U)*sizeof(double)>>>(
                device_moments.data(),paths_per_price/128U,
                paths_per_price,result,device_prices.data(),
                device_errors.data());
        check_cuda(cudaGetLastError(),"heterogeneous rough finalize");
    }
    check_cuda(cudaEventRecord(stop.get()),"heterogeneous timer stop");
    check_cuda(cudaEventSynchronize(stop.get()),"heterogeneous timer wait");
    float ms=0.0f;
    check_cuda(cudaEventElapsedTime(&ms,start.get(),stop.get()),
        "heterogeneous elapsed");
    cuda::MonteCarloRun run;
    run.prices.resize(rows.size());
    run.standard_errors.resize(rows.size());
    device_prices.copy_to(run.prices.data());
    device_errors.copy_to(run.standard_errors.data());
    run.kernel_seconds=static_cast<double>(ms)*1.0e-3;
    run.wall_seconds=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-wall_start).count();
    return run;
}

}  // namespace ai_factory::workbench::offline::pricing
