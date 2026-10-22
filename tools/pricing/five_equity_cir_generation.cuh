// Specific five-equity/CIR composition; simulation remains in the shared engine.
#pragma once

#include "pricing_rows/row_index_plan.hpp"
#include "multi_asset/rate_adjusted_model_group.cuh"
#include "correlation/dataset.hpp"
#include "model/equity/markovian/heston/dataset.hpp"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "model/equity/markovian/bates/dynamics_impl.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "model/equity/rough/hybrid_multi_asset.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "model/fixed_income/cir/correlated_dynamics.cuh"
#include "tools/datasets/multi_asset_price_dataset.hpp"
#include "tools/pricing/cross_asset_price_generation.cuh"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

using FiveEquityPack=simulation::MultiAssetPack<
    simulation::RateAdjustedModelGroup<model::equity::heston::DynamicsPolicy,1U>,
    simulation::RateAdjustedModelGroup<model::equity::bates::DynamicsPolicy,1U>,
    simulation::RateAdjustedModelGroup<model::equity::sabr::DynamicsPolicy,1U>>;
using FiveEquityRough=simulation::MultiRoughPack<
    model::equity::rough_heston::CausalFftPathPolicy,
    model::equity::rough_multi_asset::HybridPolicy<
        model::equity::rough_bergomi::PathPolicy>>;
using FiveEquityRate=model::fixed_income::cir::CorrelatedRatePolicy;

inline constexpr std::uint64_t kFiveEquitySeed=202610210700ULL;
inline constexpr std::size_t kFiveEquityPaths=1U<<20U;
inline constexpr std::size_t kFiveEquityPathChunk=8192U;
inline constexpr std::uint32_t kFiveEquityStepsPerDay=1U;

inline const std::vector<std::string>& five_equity_external_factors() {
    static const std::vector<std::string> names{
        "heston.spot","heston.variance",
        "bates.spot","bates.variance",
        "sabr.spot","sabr.volatility",
        "rough_heston.spot","rough_heston.variance",
        "rough_bergomi.spot","rough_bergomi.variance",
        "cir.rate"};
    return names;
}
inline const std::vector<std::string>& five_equity_internal_factors() {
    static const std::vector<std::string> names{
        "rough_heston.variance","rough_bergomi.variance",
        "heston.spot","heston.variance",
        "bates.spot","bates.variance",
        "sabr.spot","sabr.volatility",
        "cir.rate","rough_heston.spot","rough_bergomi.spot"};
    return names;
}

template<typename Product,unsigned TeamSize,typename LoadProduct>
int generate_five_equity_cir_prices(
    int argc,char** argv,LoadProduct load_product,
    const std::string& product_path,
    const std::string& output_path,
    const std::string& catalog_path
) {
    using Expression=RowIndexExpression;
    bool pilot=false;
    std::size_t paths=kFiveEquityPaths;
    std::size_t pilot_rows=0U;
    std::size_t product_index=0U;
    std::size_t correlation_index=0U;
    if(argc==5 && std::string_view(argv[1])=="--pilot") {
        pilot=true;
        product_index=std::stoull(argv[2]);
        correlation_index=std::stoull(argv[3]);
        paths=std::stoull(argv[4]);
        pilot_rows=4U;
    } else if(argc!=1) {
        throw std::invalid_argument(
            "Usage: generator [--pilot product_index correlation_index paths]");
    }
    const std::vector<std::filesystem::path> model_paths{
        "datasets/model/equity/markovian/heston/factor_parameters/heston_multi_core_01.json",
        "datasets/model/equity/markovian/bates/factor_parameters/bates_multi_core_01.json",
        "datasets/model/equity/markovian/sabr/factor_parameters/sabr_multi_core_01.json",
        "datasets/model/equity/rough/rough_heston/factor_parameters/rough_heston_multi_core_01.json",
        "datasets/model/equity/rough/rough_bergomi/factor_parameters/rough_bergomi_multi_core_01.json"};
    const std::string rate_path=
        "datasets/model/fixed_income/cir/factor_parameters/cir_multi_core_01.json";
    const std::string correlation_path=
        "datasets/correlation/five_equity_cir_01.json";
    MultiAssetHostModels<FiveEquityPack> equity;
    equity.models=model::equity::heston::load_models(model_paths[0]);
    equity.tail.models=model::equity::bates::load_models(model_paths[1]);
    equity.tail.tail.models=model::equity::sabr::load_models(model_paths[2]);
    MultiRoughHostModels<FiveEquityRough> rough;
    rough.models=model::equity::rough_heston::load_models(model_paths[3]);
    rough.tail.models=model::equity::rough_bergomi::load_models(model_paths[4]);
    const auto rate_models=model::fixed_income::cir::load_models(rate_path);
    auto products=load_product(product_path);
    auto correlations=correlation::load_matrices(correlation_path);
    if(equity.models.size()!=100U || equity.tail.models.size()!=100U
       || equity.tail.tail.models.size()!=100U
       || rough.models.size()!=100U || rough.tail.models.size()!=100U
       || rate_models.size()!=100U || products.size()!=100U
       || correlations.size()!=10U)
        throw std::invalid_argument("five-equity mother dataset cardinality mismatch");
    CompiledRowIndexPlan plan;
    if(pilot) {
        if(product_index>=products.size()
            || correlation_index>=correlations.size()
            || paths<128U || paths%128U!=0U)
            throw std::invalid_argument("Invalid five-equity pilot selection");
        products={products[product_index]};
        correlations={correlations[correlation_index]};
        plan=compile_row_index_plan(Expression::aligned({
            Expression::source(0U,100U),Expression::source(1U,100U),
            Expression::source(2U,100U),Expression::source(3U,100U),
            Expression::source(4U,100U),Expression::source(5U,100U),
            Expression::source(6U,1U),Expression::source(7U,1U)}),8U);
        plan.result_count=static_cast<std::uint32_t>(pilot_rows);
    } else {
        plan=compile_row_index_plan(Expression::cartesian({
            Expression::source(6U,100U),
            Expression::source(7U,10U),
            Expression::aligned({
                Expression::source(0U,100U),Expression::source(1U,100U),
                Expression::source(2U,100U),Expression::source(3U,100U),
                Expression::source(4U,100U),Expression::source(5U,100U)})
        }),8U);
        if(plan.result_count!=100000U)
            throw std::logic_error("Unexpected five-equity result count");
    }
    const auto run=execute_cross_asset_prices<
        FiveEquityPack,FiveEquityRough,FiveEquityRate,Product,TeamSize>(
        equity,rough,rate_models,products,correlations,
        five_equity_external_factors(),five_equity_internal_factors(),
        plan,paths,std::min(paths,kFiveEquityPathChunk),
        1.0f/252.0f,kFiveEquityStepsPerDay,32U,kFiveEquitySeed);
    if(pilot) {
        std::cout << "product=" << product_index
                  << " correlation=" << correlation_index
                  << " team=" << TeamSize << " rows=" << pilot_rows
                  << " paths=" << paths
                  << " kernel_seconds=" << run.kernel_seconds
                  << " wall_seconds=" << run.wall_seconds << '\n';
        for(std::size_t i=0U;i<run.prices.size();++i)
            std::cout << i << ' ' << run.prices[i] << ' '
                      << run.standard_errors[i] << '\n';
        return 0;
    }
    datasets::MultiAssetPriceDatasetRecipe recipe{
        model_paths,product_path,correlation_path,std::nullopt,
        output_path,catalog_path,"",rate_path};
    datasets::write_multi_asset_price_dataset(
        recipe,plan,run.prices,run.standard_errors,paths,kFiveEquitySeed,
        {{"engine","correlated_five_equity_cir_mc"},
         {"team_size",TeamSize},
         {"factor_order",five_equity_external_factors()},
         {"random_number_generator","Philox"},
         {"rate_scheme","CIR full-truncation Euler, trapezoid discount integral"},
         {"rough_heston_scheme","causal FFT volatility"},
         {"rough_bergomi_scheme","hybrid FFT volatility"}},
        run.wall_seconds,run.kernel_seconds);
    return 0;
}

} // namespace ai_factory::workbench::offline::pricing
