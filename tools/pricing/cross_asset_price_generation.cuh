// Offline runner for correlated equity baskets under one stochastic short rate.
#pragma once

#include "common/monte_carlo/cross_asset_kernel.cuh"
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
    const std::vector<std::uint32_t>& rate_indices,
    std::size_t scenario_count,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& external_factor_order,
    const std::vector<std::string>& internal_factor_order,
    const std::vector<monte_carlo::MultiAssetPriceRow>& rows,
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
    if(scenario_count==0U || products.empty() || correlations.empty()
        || rows.empty() || paths_per_price<2U
        || !(dt>0.0f) || steps_per_day==0U
        || external_factor_order.size()!=dimension
        || internal_factor_order.size()!=dimension
        || rate_models.empty() || rate_indices.size()!=scenario_count
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
        if(geometry.results_per_launch!=4096U
            || geometry.blocks_per_result!=0U
            || geometry.threads_per_block!=128U)
            throw std::invalid_argument(
                "rough cross-asset batch geometry is not supported");
    }
    equities.validate(scenario_count);
    rough.validate(scenario_count);
    for(const auto& rate:rate_models)
        if(!RatePolicy::valid_parameters(rate))
            throw std::invalid_argument("invalid short-rate parameters");
    for(const auto index:rate_indices)
        if(index>=rate_models.size())
            throw std::invalid_argument("short-rate index exceeds dataset");
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
    for(const auto& row:rows) {
        if(row.scenario_index>=scenario_count
            || row.product_index>=products.size()
            || row.correlation_index>=ordered.size()
            || row.discount_factor!=1.0f
            || (RatePolicy::kUsesCurve ? row.curve_index>=curves.size()
                                      : row.curve_index!=0U))
            throw std::invalid_argument("invalid cross-asset price row");
        const auto& matrix=ordered[row.correlation_index];
        equities.template validate_internal_correlation<0U,rough_count>(
            row.scenario_index,matrix);
        if constexpr (RatePolicy::kBrownianDimension==2U) {
            constexpr auto first=rough_count+EquityPack::kFactorCount;
            const auto& rate=rate_models[rate_indices[row.scenario_index]];
            double actual=0.0;
            for(std::size_t k=0U;k<=first;++k)
                actual+=static_cast<double>(matrix.lower[
                    first*(first+1U)/2U+k])
                    *static_cast<double>(matrix.lower[
                    (first+1U)*(first+2U)/2U+k]);
            if(std::abs(actual-RatePolicy::internal_correlation(rate))>2.0e-5)
                throw std::invalid_argument("rate factor correlation mismatch");
        }
        if constexpr (rough_count>0U)
            rough.template validate_internal_correlation<
                rough_count,EquityPack::kFactorCount
                    +RatePolicy::kBrownianDimension>(
                    row.scenario_index,matrix);
    }
    const auto chunk=rough_count>0U
        ? std::min(paths_per_price,path_chunk_size) : paths_per_price;
    const auto workspace_bytes=MultiRoughDeviceModels<RoughPack>::
        workspace_bytes(max_steps,chunk);
    const auto max_partials=(paths_per_price
        +(rough_count>0U ? 127U : geometry.threads_per_block-1U))
        /(rough_count>0U ? 128U : geometry.threads_per_block);
    const auto batch_rows=std::min<std::size_t>(
        geometry.results_per_launch,65535U);
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
    std::size_t moment_capacity=rough_count>0U ? max_partials : 1U;
    if constexpr (rough_count==0U)
        for(std::size_t offset=0U;offset<rows.size();offset+=batch_rows) {
            const auto count=std::min(batch_rows,rows.size()-offset);
            const auto blocks=blocks_for_count(count);
            if(blocks>1U)
                moment_capacity=std::max(moment_capacity,count*blocks);
        }
    const auto wall_start=std::chrono::steady_clock::now();
    MultiAssetDeviceModels<EquityPack> device_equities(equities);
    MultiRoughDeviceModels<RoughPack> device_rough(rough);
    cuda::DeviceBuffer<typename RatePolicy::Parameters>
        device_rates(rate_models.size());
    cuda::DeviceBuffer<std::uint32_t> device_rate_indices(rate_indices.size());
    cuda::DeviceBuffer<typename RatePolicy::CurveParameters> device_curves(
        RatePolicy::kUsesCurve ? curves.size() : 1U);
    cuda::DeviceBuffer<typename Product::ProductParameters>
        device_products(products.size());
    cuda::DeviceBuffer<monte_carlo::MultiAssetPriceRow> device_rows(rows.size());
    cuda::DeviceBuffer<float> device_factors(factors.size());
    cuda::DeviceBuffer<float> device_prices(rows.size());
    cuda::DeviceBuffer<float> device_errors(rows.size());
    cuda::DeviceBuffer<volterra::PartialMoments> device_moments(
        moment_capacity);
    cuda::DeviceBuffer<float> device_volatility(
        rough_count>0U ? rough_count*chunk*max_steps : 1U);
    cuda::DeviceBuffer<std::uint8_t> workspace(
        workspace_bytes>0U ? workspace_bytes : 1U);
    device_rates.copy_from(rate_models.data());
    device_rate_indices.copy_from(rate_indices.data());
    if constexpr (RatePolicy::kUsesCurve)
        device_curves.copy_from(curves.data());
    device_products.copy_from(products.data());
    device_rows.copy_from(rows.data());
    device_factors.copy_from(factors.data());
    const monte_carlo::CrossAssetDeviceInputs<
        EquityPack,RoughPack,RatePolicy,Product> inputs{
            device_equities.inputs(),device_rough.inputs(),
            device_rates.data(),device_rate_indices.data(),
            RatePolicy::kUsesCurve ? device_curves.data() : nullptr,
            device_products.data(),device_rows.data(),device_factors.data()};
    cuda::Event start,stop;
    check_cuda(cudaEventRecord(start.get()),"cross-asset timer start");
    if constexpr (rough_count==0U) {
        for(std::size_t offset=0U;offset<rows.size();
            offset+=batch_rows) {
            const auto count=std::min(batch_rows,rows.size()-offset);
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
                volterra::finalize_price_batch_kernel
                    <<<static_cast<unsigned>(count),128U,
                        8U*sizeof(double)>>>(
                        device_moments.data(),blocks,paths_per_price,
                        offset,device_prices.data(),device_errors.data());
                check_cuda(cudaGetLastError(),
                    "cross-asset batch finalize");
            }
        }
    } else {
    for(std::size_t result=0U;result<rows.size();++result) {
        const auto steps=products[rows[result].product_index].maturity_days
            *steps_per_day;
        for(std::size_t offset=0U;offset<paths_per_price;offset+=chunk) {
            const auto count=std::min(chunk,paths_per_price-offset);
            if constexpr (rough_count>0U)
                device_rough.template launch_volatilities<
                    dimension,EquityPack::kAssetCount>(
                        device_rows.data(),device_factors.data(),result,
                        offset,count,dt,steps,direct_tile_size,base_seed,
                        workspace.data(),workspace.size(),
                        device_volatility.data());
            monte_carlo::launch_cross_asset_cuda<
                EquityPack,RoughPack,RatePolicy,Product,false,TeamSize>(
                    inputs,result,offset,count,steps,steps_per_day,dt,
                    base_seed,rough_count>0U ? device_volatility.data():nullptr,
                    device_moments.data());
        }
        volterra::finalize_price_kernel<<<1U,128U,
            8U*sizeof(double)>>>(
                device_moments.data(),(paths_per_price+127U)/128U,
                paths_per_price,result,device_prices.data(),
                device_errors.data());
        check_cuda(cudaGetLastError(),"cross-asset finalize");
    }
    }
    check_cuda(cudaEventRecord(stop.get()),"cross-asset timer stop");
    check_cuda(cudaEventSynchronize(stop.get()),"cross-asset timer wait");
    float ms=0.0f;
    check_cuda(cudaEventElapsedTime(&ms,start.get(),stop.get()),
        "cross-asset elapsed");
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
