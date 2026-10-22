// Static rough-type composition for volatility-first multi-asset pricing.
#pragma once

#include "common/monte_carlo/multi_asset_heterogeneous_rough_kernel.cuh"
#include "common/volterra/causal_fft_volatility_kernel.cuh"
#include "common/volterra/hybrid_fft_volatility_kernel.cuh"
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

inline std::size_t choose_rough_price_batch_size(
    std::size_t row_count,std::size_t requested,
    std::size_t bytes_per_row
) {
    if(requested==0U) throw std::invalid_argument(
        "rough price batch size must be positive");
    std::size_t free_bytes=0U,total_bytes=0U;
    check_cuda(cudaMemGetInfo(&free_bytes,&total_bytes),
        "rough price memory budget");
    const std::size_t affordable=bytes_per_row==0U
        ? requested : std::max<std::size_t>(
            1U,(free_bytes/2U)/bytes_per_row);
    return std::min({row_count,requested,affordable});
}

template<typename Pack>
struct MultiRoughHostModels;

template<>
struct MultiRoughHostModels<simulation::MultiRoughPack<>> {
    void validate(const std::vector<RowIndexSelector>&,
                          std::size_t) const {}
};

template<typename Head,typename... Tail>
struct MultiRoughHostModels<simulation::MultiRoughPack<Head,Tail...>> {
    using Rest=simulation::MultiRoughPack<Tail...>;
    std::vector<typename Head::Parameters> models;
    MultiRoughHostModels<Rest> tail;

    void validate(
        const std::vector<RowIndexSelector>& selectors,
        std::size_t asset_offset
    ) const {
        if(models.empty() || selectors.size()<=asset_offset
            || selectors[asset_offset].source_count!=models.size())
            throw std::invalid_argument("invalid compact rough model slot");
        tail.validate(selectors,asset_offset+1U);
    }
};

template<typename Policy, std::size_t Count>
struct MultiRoughHostModels<
    simulation::HomogeneousRoughPack<Policy, Count>> {
    std::vector<typename Policy::Parameters> models;

    void validate(
        const std::vector<RowIndexSelector>& selectors,
        std::size_t asset_offset
    ) const {
        if(models.empty() || selectors.size()<asset_offset+Count)
            throw std::invalid_argument("invalid compact rough model slots");
        for(std::size_t asset=0U;asset<Count;++asset)
            if(selectors[asset_offset+asset].source_count!=models.size())
                throw std::invalid_argument(
                    "compact rough selector disagrees with dataset");
    }
};

template<typename Pack>
struct MultiRoughDeviceModels;

template<>
struct MultiRoughDeviceModels<simulation::MultiRoughPack<>> {
    explicit MultiRoughDeviceModels(
        const MultiRoughHostModels<simulation::MultiRoughPack<>>&
    ) {}
    simulation::MultiRoughPack<>::Inputs inputs(
        const RowIndexSelector* =nullptr,std::size_t=0U
    ) const { return {}; }
    static std::size_t workspace_bytes(std::uint32_t,std::size_t) {
        return 0U;
    }
};

template<typename Head,typename... Tail>
struct MultiRoughDeviceModels<simulation::MultiRoughPack<Head,Tail...>> {
    using Pack=simulation::MultiRoughPack<Head,Tail...>;
    using Rest=simulation::MultiRoughPack<Tail...>;
    cuda::DeviceBuffer<typename Head::Parameters> models;
    MultiRoughDeviceModels<Rest> tail;

    explicit MultiRoughDeviceModels(const MultiRoughHostModels<Pack>& host)
        :models(host.models.size()),tail(host.tail) {
        models.copy_from(host.models.data());
    }
    typename Pack::Inputs inputs(
        const RowIndexSelector* selectors,
        std::size_t asset_offset=0U
    ) const {
        return {models.data(),tail.inputs(selectors,asset_offset+1U),
                selectors+asset_offset};
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
        const float* packed_factors,
        std::size_t result_index,std::size_t path_offset,
        std::size_t path_count,float dt,std::uint32_t steps,
        std::uint32_t direct_tile_size,std::uint64_t seed,
        void* workspace,std::size_t workspace_bytes,
        float* volatility_paths,cudaStream_t stream,
        const RowIndexSelector* row_selectors,
        std::size_t correlation_slot
    ) const {
        constexpr std::uint32_t source=static_cast<std::uint32_t>(
            2U+MarkovAssets+AssetOffset);
        float* output=volatility_paths+AssetOffset*path_count*steps;
        if constexpr (requires { typename Head::VolatilityState; }) {
            volterra::causal_fft::launch_volatility_factor_cuda<
                Head,Dimension>(
                    models.data(),packed_factors,
                    result_index,path_offset,path_count,dt,steps,
                    direct_tile_size,seed,workspace,workspace_bytes,
                    output,AssetOffset,source,stream,
                    row_selectors,MarkovAssets+AssetOffset,correlation_slot);
        } else {
            volterra::hybrid_fft::launch_correlated_hybrid_volatility_cuda<
                Head,Dimension>(
                    models.data(),packed_factors,
                    result_index,path_offset,path_count,dt,steps,seed,
                    workspace,workspace_bytes,output,AssetOffset,source,
                    stream,row_selectors,
                    MarkovAssets+AssetOffset,correlation_slot);
        }
        if constexpr (Rest::kAssetCount>0U)
            tail.template launch_volatilities<
                Dimension,MarkovAssets,AssetOffset+1U>(
                    packed_factors,result_index,path_offset,
                    path_count,dt,steps,direct_tile_size,seed,
                    workspace,workspace_bytes,volatility_paths,stream,
                    row_selectors,correlation_slot);
    }
};

template<typename Policy, std::size_t Count>
struct MultiRoughDeviceModels<
    simulation::HomogeneousRoughPack<Policy, Count>> {
    using Pack = simulation::HomogeneousRoughPack<Policy, Count>;
    cuda::DeviceBuffer<typename Policy::Parameters> models;

    explicit MultiRoughDeviceModels(
        const MultiRoughHostModels<Pack>& host
    ) : models(host.models.size()) {
        models.copy_from(host.models.data());
    }

    typename Pack::Inputs inputs(
        const RowIndexSelector* selectors,
        std::size_t asset_offset=0U
    ) const {
        return {models.data(),selectors+asset_offset};
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
        const float* packed_factors,
        std::size_t result_index, std::size_t path_offset,
        std::size_t path_count, float dt, std::uint32_t steps,
        std::uint32_t direct_tile_size, std::uint64_t seed,
        void* workspace, std::size_t workspace_bytes,
        float* volatility_paths,cudaStream_t stream,
        const RowIndexSelector* row_selectors,
        std::size_t correlation_slot
    ) const {
        for (std::size_t asset = 0U; asset < Count; ++asset) {
            float* output = volatility_paths
                + asset * path_count * steps;
            const auto source = static_cast<std::uint32_t>(
                2U + MarkovAssets + asset);
            if constexpr (Pack::kAllCausal) {
                volterra::causal_fft::launch_volatility_factor_cuda<
                    Policy, Dimension>(
                        models.data(),packed_factors, result_index, path_offset,
                        path_count, dt, steps, direct_tile_size, seed,
                        workspace, workspace_bytes, output,
                        asset, source, stream,
                        row_selectors,MarkovAssets+asset,correlation_slot);
            } else {
                volterra::hybrid_fft::
                    launch_correlated_hybrid_volatility_cuda<
                        Policy, Dimension>(
                            models.data(),packed_factors, result_index, path_offset,
                            path_count, dt, steps, seed, workspace,
                            workspace_bytes, output, asset, source,
                            stream,row_selectors,
                            MarkovAssets+asset,correlation_slot);
            }
        }
    }
};

template<typename MarkovPack,typename RoughPack,typename Product,
         unsigned TeamSize=1U>
cuda::MonteCarloRun execute_heterogeneous_rough_prices(
    const MultiAssetHostModels<MarkovPack>& markov_models,
    const MultiRoughHostModels<RoughPack>& rough_models,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& external_factor_order,
    const std::vector<std::string>& internal_factor_order,
    const CompiledRowIndexPlan& row_plan,
    std::size_t paths_per_price,std::size_t path_chunk_size,
    float dt,std::uint32_t steps_per_day,std::uint32_t direct_tile_size,
    std::uint64_t base_seed,std::size_t price_batch_size=4U
) {
    static_assert(RoughPack::kAssetCount>0U);
    static_assert(MarkovPack::kAssetCount+RoughPack::kAssetCount<=254U);
    constexpr std::size_t dimension=RoughPack::kAssetCount
        +MarkovPack::kFactorCount+RoughPack::kTwoFactorCount;
    constexpr std::size_t lower_count=dimension*(dimension+1U)/2U;
    const std::size_t result_count=row_plan.result_count;
    if(row_plan.selectors.size()!=MarkovPack::kAssetCount
            +RoughPack::kAssetCount+2U
        || products.empty() || correlations.empty()
        || result_count==0U || paths_per_price<128U
        || paths_per_price%128U!=0U || path_chunk_size==0U
        || path_chunk_size%128U!=0U || !(dt>0.0f)
        || steps_per_day==0U
        || (direct_tile_size!=32U && direct_tile_size!=64U)
        || external_factor_order.size()!=dimension
        || internal_factor_order.size()!=dimension)
        throw std::invalid_argument("invalid heterogeneous rough inputs");
    markov_models.validate(row_plan.selectors,0U);
    rough_models.validate(row_plan.selectors,
        MarkovPack::kAssetCount);
    constexpr std::size_t product_slot=
        MarkovPack::kAssetCount+RoughPack::kAssetCount;
    if(row_plan.selectors[product_slot].source_count!=products.size()
        || row_plan.selectors[product_slot+1U].source_count
            !=correlations.size())
        throw std::invalid_argument(
            "compact rough selector disagrees with dataset");
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
    const auto chunk=std::min(paths_per_price,path_chunk_size);
    const auto workspace_bytes=MultiRoughDeviceModels<RoughPack>::
        workspace_bytes(max_steps,chunk);
    const std::size_t partial_count=paths_per_price/128U;
    const std::size_t volatility_values=
        RoughPack::kAssetCount*chunk*max_steps;
    const std::size_t bytes_per_price=workspace_bytes
        +volatility_values*sizeof(float)
        +partial_count*sizeof(reductions::MomentSums);
    const auto batch_size=choose_rough_price_batch_size(
        result_count,price_batch_size,bytes_per_price);
    const auto wall_start=std::chrono::steady_clock::now();
    MultiAssetDeviceModels<MarkovPack> device_markov(markov_models);
    MultiRoughDeviceModels<RoughPack> device_rough(rough_models);
    cuda::DeviceBuffer<typename Product::ProductParameters>
        device_products(products.size());
    cuda::DeviceBuffer<RowIndexSelector> device_selectors(
        row_plan.selectors.size());
    cuda::DeviceBuffer<float> device_factors(factors.size());
    cuda::DeviceBuffer<float> device_prices(result_count);
    cuda::DeviceBuffer<float> device_errors(result_count);
    cuda::DeviceBuffer<reductions::MomentSums>
        device_moments(batch_size*partial_count);
    cuda::DeviceBuffer<float> device_volatility(
        batch_size*volatility_values);
    cuda::DeviceBuffer<std::uint8_t> workspace(
        batch_size*workspace_bytes);
    device_products.copy_from(products.data());
    device_selectors.copy_from(row_plan.selectors.data());
    device_factors.copy_from(factors.data());
    const monte_carlo::HeterogeneousRoughDeviceInputs<
        MarkovPack,RoughPack,Product> inputs{
            device_markov.inputs(device_selectors.data()),
            device_rough.inputs(device_selectors.data(),
                MarkovPack::kAssetCount),device_products.data(),
            device_factors.data(),device_selectors.data()};
    const auto launch_price=[&](std::size_t result,std::size_t slot,
                                cudaStream_t stream) {
        constexpr std::size_t product_slot=
            MarkovPack::kAssetCount+RoughPack::kAssetCount;
        const auto product_index=row_plan.selectors[product_slot].select(
            static_cast<std::uint32_t>(result));
        const auto steps=products[product_index].maturity_days
            *steps_per_day;
        auto* row_workspace=workspace.data()+slot*workspace_bytes;
        auto* row_volatility=device_volatility.data()
            +slot*volatility_values;
        auto* row_moments=device_moments.data()+slot*partial_count;
        for(std::size_t offset=0U;offset<paths_per_price;offset+=chunk) {
            const auto count=std::min(chunk,paths_per_price-offset);
            device_rough.template launch_volatilities<
                dimension,MarkovPack::kAssetCount>(
                    device_factors.data(),result,offset,count,dt,steps,
                    direct_tile_size,base_seed,row_workspace,workspace_bytes,
                    row_volatility,stream,
                    device_selectors.data(),
                    product_slot+1U);
            monte_carlo::launch_heterogeneous_rough_cuda<
                MarkovPack,RoughPack,Product,false,TeamSize>(
                    inputs,result,offset,count,steps,steps_per_day,dt,
                    base_seed,row_volatility,row_moments,nullptr,stream);
        }
        monte_carlo::finalize_price_kernel<<<1U,128U,
            2U*(128U/32U)*sizeof(double),stream>>>(
                row_moments,partial_count,paths_per_price,result,
                device_prices.data(),device_errors.data());
        check_cuda(cudaGetLastError(),"heterogeneous rough finalize");
    };
    cuda::Event start,stop;
    check_cuda(cudaEventRecord(start.get()),"heterogeneous timer start");
    if(batch_size==1U) {
        for(std::size_t result=0U;result<result_count;++result)
            launch_price(result,0U,0);
    } else {
        std::vector<cuda::Stream> streams(batch_size);
        std::vector<cuda::Event> completed(batch_size);
        for(auto& stream:streams)
            check_cuda(cudaStreamWaitEvent(stream.get(),start.get()),
                "heterogeneous batch start dependency");
        for(std::size_t result=0U;result<result_count;++result)
            launch_price(result,result%batch_size,
                streams[result%batch_size].get());
        for(std::size_t slot=0U;slot<batch_size;++slot) {
            check_cuda(cudaEventRecord(
                completed[slot].get(),streams[slot].get()),
                "heterogeneous row completion");
            check_cuda(cudaStreamWaitEvent(0,completed[slot].get()),
                "heterogeneous batch completion");
        }
    }
    check_cuda(cudaEventRecord(stop.get()),"heterogeneous timer stop");
    check_cuda(cudaEventSynchronize(stop.get()),"heterogeneous timer wait");
    float ms=0.0f;
    check_cuda(cudaEventElapsedTime(&ms,start.get(),stop.get()),
        "heterogeneous elapsed");
    cuda::MonteCarloRun run;
    run.prices.resize(result_count);
    run.standard_errors.resize(result_count);
    device_prices.copy_to(run.prices.data());
    device_errors.copy_to(run.standard_errors.data());
    run.kernel_seconds=static_cast<double>(ms)*1.0e-3;
    run.wall_seconds=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-wall_start).count();
    return run;
}

}  // namespace ai_factory::workbench::offline::pricing
