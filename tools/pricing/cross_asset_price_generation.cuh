// Offline runner for correlated equity baskets under one short-rate policy.
#pragma once

#include "multi_asset/equity_rate_price_kernel.cuh"
#include "tools/pricing/heterogeneous_rough_price_generation.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

struct CrossAssetBatchGeometry {
    std::size_t results_per_launch=4096U;
    unsigned blocks_per_result=0U; // Zero selects from SM count and path work.
    unsigned threads_per_block=128U;
};

template<typename EquityPack,typename RoughPack,typename RatePolicy,
         typename Product,unsigned TeamSize=1U>
cuda::MonteCarloRun execute_cross_asset_prices(
    const MultiAssetHostModels<EquityPack>& equities,
    const MultiRoughHostModels<RoughPack>& rough,
    const std::vector<typename RatePolicy::Parameters>& rate_models,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& external_factor_order,
    const std::vector<std::string>& internal_factor_order,
    const CompiledRowIndexPlan& row_plan,
    std::size_t paths_per_price,std::size_t path_chunk_size,
    float dt,std::uint32_t steps_per_day,std::uint32_t direct_tile_size,
    std::uint64_t base_seed,
    const std::vector<typename RatePolicy::CurveParameters>& curves={},
    CrossAssetBatchGeometry geometry={}
) {
    static_assert(EquityPack::kAssetCount+RoughPack::kAssetCount>0U);
    static_assert(Product::kTerminalCashflow);
    constexpr std::size_t rough_count=RoughPack::kAssetCount;
    constexpr std::size_t dimension=rough_count
        +EquityPack::kFactorCount+RatePolicy::kBrownianDimension
        +RoughPack::kTwoFactorCount;
    constexpr std::size_t lower_count=dimension*(dimension+1U)/2U;
    const std::size_t result_count=row_plan.result_count;
    if(row_plan.selectors.size()!=EquityPack::kAssetCount
            +RoughPack::kAssetCount+3U
            +(RatePolicy::kUsesCurve ? 1U : 0U)
        || products.empty() || correlations.empty()
        || result_count==0U || paths_per_price<2U
        || !(dt>0.0f) || steps_per_day==0U
        || external_factor_order.size()!=dimension
        || internal_factor_order.size()!=dimension
        || rate_models.empty()
        || (RatePolicy::kUsesCurve && curves.empty())
        || geometry.results_per_launch==0U
        || geometry.threads_per_block<32U
        || geometry.threads_per_block>1024U
        || geometry.threads_per_block%32U!=0U
        || geometry.threads_per_block%TeamSize!=0U)
        throw std::invalid_argument("invalid cross-asset inputs");
    if constexpr (rough_count>0U) {
        if(paths_per_price<128U || paths_per_price%128U!=0U
            || path_chunk_size==0U
            || path_chunk_size%128U!=0U
            || (direct_tile_size!=32U && direct_tile_size!=64U))
            throw std::invalid_argument("invalid cross-asset FFT geometry");
        if(geometry.blocks_per_result!=0U
            || geometry.threads_per_block!=128U)
            throw std::invalid_argument(
                "rough cross-asset batch geometry is not supported");
    }
    equities.validate(row_plan.selectors,0U);
    rough.validate(row_plan.selectors,
        EquityPack::kAssetCount);
    const auto rate_slot=EquityPack::kAssetCount+rough_count;
    if(row_plan.selectors[rate_slot].source_count!=rate_models.size()
        || row_plan.selectors[rate_slot+1U].source_count!=products.size()
        || row_plan.selectors[rate_slot+2U].source_count!=correlations.size()
        || (RatePolicy::kUsesCurve
            && row_plan.selectors[rate_slot+3U].source_count
                !=curves.size()))
        throw std::invalid_argument(
            "compact cross-asset selector disagrees with dataset");
    for(const auto& rate:rate_models)
        if(!RatePolicy::valid_parameters(rate))
            throw std::invalid_argument("invalid short-rate parameters");
    if constexpr (RatePolicy::kUsesCurve)
        for(const auto& curve:curves)
            if(!RatePolicy::valid_curve(curve))
                throw std::invalid_argument("invalid initial curve");
    std::vector<std::string> variance_names,spot_names;
    if constexpr (rough_count>0U)
        RoughPack::append_factor_names(external_factor_order,
            EquityPack::kFactorCount,variance_names,spot_names);
    std::vector<std::string> required=variance_names;
    required.insert(required.end(),external_factor_order.begin(),
        external_factor_order.begin()+EquityPack::kFactorCount);
    required.insert(required.end(),external_factor_order.end()
        -RatePolicy::kBrownianDimension,external_factor_order.end());
    required.insert(required.end(),spot_names.begin(),spot_names.end());
    if(internal_factor_order!=required)
        throw std::invalid_argument("cross-asset factor order mismatch");
    std::vector<correlation::MatrixRow> ordered;
    std::vector<float> factors;
    ordered.reserve(correlations.size());
    factors.reserve(correlations.size()*lower_count);
    for(const auto& matrix:correlations) {
        if(matrix.factors!=external_factor_order
            || matrix.lower.size()!=lower_count)
            throw std::invalid_argument("cross-asset external factors mismatch");
        ordered.push_back(correlation::reorder_factors(
            matrix,internal_factor_order));
        factors.insert(factors.end(),ordered.back().lower.begin(),
            ordered.back().lower.end());
    }
    std::uint32_t max_steps=0U;
    for(const auto& product:products) {
        if constexpr (requires { Product::valid_parameters(
            product,steps_per_day); })
            if(!Product::valid_parameters(product,steps_per_day))
                throw std::invalid_argument("invalid cross-asset product");
        const auto count=static_cast<std::uint64_t>(
            product.maturity_days)*steps_per_day;
        const auto limit=rough_count==0U
            ? philox::kMaximumDomainStep
            : (RoughPack::kAllCausal ? 8192U : 4096U);
        if(count==0U || count>limit)
            throw std::invalid_argument("cross-asset maturity exceeds capacity");
        max_steps=std::max(max_steps,static_cast<std::uint32_t>(count));
    }
    const auto chunk=rough_count>0U
        ? std::min(paths_per_price,path_chunk_size) : paths_per_price;
    const auto workspace_bytes=MultiRoughDeviceModels<RoughPack>::
        workspace_bytes(max_steps,chunk);
    const std::size_t volatility_values=
        rough_count*chunk*max_steps;
    const auto max_partials=(paths_per_price
        +(rough_count>0U ? 127U : geometry.threads_per_block-1U))
        /(rough_count>0U ? 128U : geometry.threads_per_block);
    const auto batch_rows=std::min<std::size_t>(
        geometry.results_per_launch,65535U);
    const auto rough_batch_size=rough_count>0U
        ? choose_rough_price_batch_size(
            result_count,std::min<std::size_t>(
                geometry.results_per_launch,4U),
            workspace_bytes+volatility_values*sizeof(float)
                +max_partials*sizeof(reductions::MomentSums))
        : 1U;
    cudaDeviceProp properties{};
    if constexpr (rough_count==0U) {
        int device=0;
        check_cuda(cudaGetDevice(&device),"cross-asset batch device");
        check_cuda(cudaGetDeviceProperties(&properties,device),
            "cross-asset batch properties");
    }
    const auto blocks_for_count=[&](std::size_t count)->unsigned {
        const auto available=static_cast<unsigned>(max_partials);
        if(geometry.blocks_per_result>0U)
            return std::min(available,geometry.blocks_per_result);
        if(count==1U) return available;
        const unsigned occupancy=static_cast<unsigned>(
            (2U*properties.multiProcessorCount+count-1U)/count);
        const unsigned tiles_per_block=dimension>=16U ? 2U : 4U;
        const unsigned work=static_cast<unsigned>(
            (max_partials+tiles_per_block-1U)/tiles_per_block);
        return std::min(available,std::max({1U,occupancy,work}));
    };
    std::size_t moment_capacity=rough_count>0U
        ? max_partials*rough_batch_size : 1U;
    if constexpr (rough_count==0U)
        for(std::size_t offset=0U;offset<result_count;offset+=batch_rows) {
            const auto count=std::min(batch_rows,result_count-offset);
            const auto blocks=blocks_for_count(count);
            if(blocks>1U)
                moment_capacity=std::max(moment_capacity,count*blocks);
        }
    const auto wall_start=std::chrono::steady_clock::now();
    MultiAssetDeviceModels<EquityPack> device_equities(equities);
    MultiRoughDeviceModels<RoughPack> device_rough(rough);
    cuda::DeviceBuffer<typename RatePolicy::Parameters>
        device_rates(rate_models.size());
    cuda::DeviceBuffer<RowIndexSelector> device_selectors(
        row_plan.selectors.size());
    cuda::DeviceBuffer<typename RatePolicy::CurveParameters> device_curves(
        RatePolicy::kUsesCurve ? curves.size() : 1U);
    cuda::DeviceBuffer<typename Product::ProductParameters>
        device_products(products.size());
    cuda::DeviceBuffer<float> device_factors(factors.size());
    cuda::DeviceBuffer<float> device_prices(result_count);
    cuda::DeviceBuffer<float> device_errors(result_count);
    cuda::DeviceBuffer<reductions::MomentSums> device_moments(
        moment_capacity);
    cuda::DeviceBuffer<float> device_volatility(
        rough_count>0U ? rough_batch_size*volatility_values : 1U);
    cuda::DeviceBuffer<std::uint8_t> workspace(
        workspace_bytes>0U ? rough_batch_size*workspace_bytes : 1U);
    device_rates.copy_from(rate_models.data());
    device_selectors.copy_from(row_plan.selectors.data());
    if constexpr (RatePolicy::kUsesCurve)
        device_curves.copy_from(curves.data());
    device_products.copy_from(products.data());
    device_factors.copy_from(factors.data());
    const monte_carlo::CrossAssetDeviceInputs<
        EquityPack,RoughPack,RatePolicy,Product> inputs{
            device_equities.inputs(device_selectors.data()),
            device_rough.inputs(device_selectors.data(),
                EquityPack::kAssetCount),
            device_rates.data(),
            RatePolicy::kUsesCurve ? device_curves.data() : nullptr,
            device_products.data(),device_factors.data(),
            device_selectors.data()};
    cuda::Event start,stop;
    check_cuda(cudaEventRecord(start.get()),"cross-asset timer start");
    if constexpr (rough_count==0U) {
        for(std::size_t offset=0U;offset<result_count;
            offset+=batch_rows) {
            const auto count=std::min(batch_rows,result_count-offset);
            const unsigned blocks=blocks_for_count(count);
            monte_carlo::launch_cross_asset_batch_cuda<
                EquityPack,RoughPack,RatePolicy,Product,false,TeamSize>(
                    inputs,offset,count,0U,paths_per_price,
                    blocks,blocks,geometry.threads_per_block,
                    steps_per_day,dt,base_seed,nullptr,
                    device_moments.data(),nullptr,
                    blocks==1U ? device_prices.data() : nullptr,
                    blocks==1U ? device_errors.data() : nullptr);
            if(blocks>1U) {
                monte_carlo::finalize_price_batch_kernel
                    <<<static_cast<unsigned>(count),128U,
                        8U*sizeof(double)>>>(
                        device_moments.data(),blocks,paths_per_price,
                        offset,device_prices.data(),device_errors.data());
                check_cuda(cudaGetLastError(),
                    "cross-asset batch finalize");
            }
        }
    } else {
        const auto launch_price=[&](std::size_t result,std::size_t slot,
                                    cudaStream_t stream) {
            const auto product_index=
                row_plan.selectors[EquityPack::kAssetCount+rough_count+1U]
                    .select(static_cast<std::uint32_t>(result));
            const auto steps=products[product_index]
                .maturity_days*steps_per_day;
            auto* row_workspace=workspace.data()+slot*workspace_bytes;
            auto* row_volatility=device_volatility.data()
                +slot*volatility_values;
            auto* row_moments=device_moments.data()+slot*max_partials;
            for(std::size_t offset=0U;offset<paths_per_price;
                offset+=chunk) {
                const auto count=std::min(chunk,
                    paths_per_price-offset);
                device_rough.template launch_volatilities<
                    dimension,EquityPack::kAssetCount>(
                        device_factors.data(),result,offset,count,dt,steps,
                        direct_tile_size,base_seed,row_workspace,
                        workspace_bytes,row_volatility,stream,
                        device_selectors.data(),
                        EquityPack::kAssetCount+rough_count+2U);
                monte_carlo::launch_cross_asset_cuda<
                    EquityPack,RoughPack,RatePolicy,Product,false,TeamSize>(
                        inputs,result,offset,count,steps,steps_per_day,dt,
                        base_seed,row_volatility,row_moments,nullptr,stream);
            }
            monte_carlo::finalize_price_kernel<<<1U,128U,
                8U*sizeof(double),stream>>>(
                    row_moments,max_partials,paths_per_price,result,
                    device_prices.data(),device_errors.data());
            check_cuda(cudaGetLastError(),"cross-asset finalize");
        };
        if(rough_batch_size==1U) {
            for(std::size_t result=0U;result<result_count;++result)
                launch_price(result,0U,0);
        } else {
            std::vector<cuda::Stream> streams(rough_batch_size);
            std::vector<cuda::Event> completed(rough_batch_size);
            for(auto& stream:streams)
                check_cuda(cudaStreamWaitEvent(stream.get(),start.get()),
                    "cross-asset rough batch start dependency");
            for(std::size_t result=0U;result<result_count;++result)
                launch_price(result,result%rough_batch_size,
                    streams[result%rough_batch_size].get());
            for(std::size_t slot=0U;slot<rough_batch_size;++slot) {
                check_cuda(cudaEventRecord(
                    completed[slot].get(),streams[slot].get()),
                    "cross-asset rough row completion");
                check_cuda(cudaStreamWaitEvent(0,completed[slot].get()),
                    "cross-asset rough batch completion");
            }
        }
    }
    check_cuda(cudaEventRecord(stop.get()),"cross-asset timer stop");
    check_cuda(cudaEventSynchronize(stop.get()),"cross-asset timer wait");
    float ms=0.0f;
    check_cuda(cudaEventElapsedTime(&ms,start.get(),stop.get()),
        "cross-asset elapsed");
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
