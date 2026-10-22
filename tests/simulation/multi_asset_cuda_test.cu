#include "correlation/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/markovian/stein_stein/dynamics_impl.cuh"
#include "model/equity/markovian/schobel_zhu/dynamics_impl.cuh"
#include "model/equity/markovian/heston_3_2/dynamics_impl.cuh"
#include "model/equity/markovian/kou/dynamics_impl.cuh"
#include "model/equity/markovian/merton/dynamics_impl.cuh"
#include "model/equity/markovian/cev/dynamics_impl.cuh"
#include "model/equity/markovian/bates/dynamics_impl.cuh"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "product/best_of_call/pricing_policy.cuh"
#include "product/basket_call/pricing_policy.cuh"
#include "tools/pricing/multi_asset_price_generation.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wb=ai_factory::workbench;
using Expression=wb::RowIndexExpression;
using BS=wb::model::equity::black_scholes::DynamicsPolicy;
using Heston=wb::model::equity::heston::DynamicsPolicy;
using SABR=wb::model::equity::sabr::DynamicsPolicy;
using Worst=wb::product::WorstOfCallPathPolicy;

void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}

wb::offline::pricing::BatchedMonteCarloProfile profile(std::size_t paths) {
    wb::offline::pricing::BatchedMonteCarloProfile p{};
    p.paths_per_price=paths;
    p.results_per_launch=128U;
    p.block_count_limit=128U;
    p.threads_per_block=128U;
    p.seed=19433U;
    return p;
}

void check_cholesky() {
    bool rejected=false;
    try {
        wb::correlation::packed_cholesky({1.0,0.9,0.9,
            0.9,1.0,-0.9,0.9,-0.9,1.0},3U);
    } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"indefinite correlation matrix accepted");
}

void check_thirty_assets() {
    using Pack=wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<BS,30U>>;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models={{1.0f,0.0f,0.0f,0.0f}};
    std::vector<std::string> names;
    std::vector<double> dense(900U,0.0);
    for(std::size_t i=0U;i<30U;++i) {
        names.push_back("asset_"+std::to_string(i)+".spot");
        dense[i*30U+i]=1.0;
    }
    const std::vector<wb::correlation::MatrixRow> matrices{{
        "identity",names,wb::correlation::packed_cholesky(dense,30U)}};
    std::vector<Expression> slots;
    for(std::size_t i=0U;i<32U;++i) slots.push_back(Expression::source(i,1U));
    const auto one=wb::compile_row_index_plan(Expression::aligned(slots),32U);
    const auto deterministic=wb::offline::pricing::execute_multi_asset_prices<
        Pack,Worst,4U>(host,{{0.0f,1U}},matrices,names,one,
            profile(256U),1.0f/252.0f,1U);
    require(deterministic.prices[0]==1.0f
        &&deterministic.standard_errors[0]==0.0f,
        "thirty deterministic assets priced incorrectly");
    host.models[0].volatility=0.2f;
    const std::vector<Worst::ProductParameters> products{{0.7f,21U}};
    const auto scalar=wb::offline::pricing::execute_multi_asset_prices<
        Pack,Worst,1U>(host,products,matrices,names,one,
            profile(8192U),1.0f/252.0f,1U);
    const auto four=wb::offline::pricing::execute_multi_asset_prices<
        Pack,Worst,4U>(host,products,matrices,names,one,
            profile(8192U),1.0f/252.0f,1U);
    require(scalar.prices==four.prices
        &&scalar.standard_errors==four.standard_errors,
        "path team size changed thirty asset price");
    std::cout<<"D=30: TeamSize=1 kernel "<<scalar.kernel_seconds
             <<" s; TeamSize=4 kernel "<<four.kernel_seconds<<" s\n";
    auto many=slots;
    many[30U]=Expression::source(30U,100U);
    const auto hundred=wb::compile_row_index_plan(
        Expression::cartesian({
            Expression::aligned(std::vector<Expression>(many.begin(),many.begin()+30U)),
            many[30U],many[31U]}),32U);
    const std::vector<Worst::ProductParameters> hundred_products(100U,{0.7f,21U});
    auto p=profile(512U);
    const auto single_block=wb::offline::pricing::execute_multi_asset_prices<
        Pack,Worst,4U>(host,hundred_products,matrices,names,hundred,p,
            1.0f/252.0f,1U);
    p.blocks_per_price=4U;
    const auto multi_block=wb::offline::pricing::execute_multi_asset_prices<
        Pack,Worst,4U>(host,hundred_products,matrices,names,hundred,p,
            1.0f/252.0f,1U);
    require(single_block.prices.size()==100U,"compact row count incorrect");
    for(std::size_t i=0U;i<100U;++i)
        require(std::abs(single_block.prices[i]-multi_block.prices[i])<2e-6f,
            "multi block price differs");
}

void check_stochastic_volatility() {
    using Pack=wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<BS,1U>,
        wb::simulation::ModelGroup<Heston,1U>,
        wb::simulation::ModelGroup<SABR,1U>>;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models={{1.0f,0.0f,0.0f,0.2f}};
    host.tail.models={{1.0f,0.0f,0.0f,0.04f,
        2.0f,0.04f,0.5f,-0.6f}};
    host.tail.tail.models={{1.0f,0.0f,0.0f,0.2f,
        0.4f,-0.3f,1.0f}};
    const std::vector<std::string> names{
        "bs.spot","heston.spot","heston.variance",
        "sabr.spot","sabr.volatility"};
    const std::vector<double> dense{
        1.0,0.2,0.0,0.0,0.0,
        0.2,1.0,-0.6,0.0,0.0,
        0.0,-0.6,1.0,0.0,0.0,
        0.0,0.0,0.0,1.0,-0.3,
        0.0,0.0,0.0,-0.3,1.0};
    const std::vector<wb::correlation::MatrixRow> matrices{{
        "mixed",names,wb::correlation::packed_cholesky(dense,5U)}};
    const auto plan=wb::compile_row_index_plan(Expression::aligned({
        Expression::source(0U,1U),Expression::source(1U,1U),
        Expression::source(2U,1U),Expression::source(3U,1U),
        Expression::source(4U,1U)}),5U);
    const auto p=profile(1024U);
    const auto one=wb::offline::pricing::execute_multi_asset_prices<
        Pack,Worst,1U>(host,{{0.8f,21U}},matrices,names,plan,p,
            1.0f/252.0f,1U);
    const auto four=wb::offline::pricing::execute_multi_asset_prices<
        Pack,Worst,4U>(host,{{0.8f,21U}},matrices,names,plan,p,
            1.0f/252.0f,1U);
    require(std::isfinite(one.prices[0]) && one.prices[0]>=0.0f,
        "invalid stochastic-volatility basket price");
    require(one.prices==four.prices
        &&one.standard_errors==four.standard_errors,
        "stochastic-volatility price depends on team size");
}

void check_auxiliary_jump_stream() {
    using CEV = wb::model::equity::cev::DynamicsPolicy;
    using Bates = wb::model::equity::bates::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<CEV, 1U>,
        wb::simulation::ModelGroup<Bates, 1U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models.push_back({1.0f, 0.0f, 0.0f, 0.2f, 0.8f});
    host.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.04f,
                                2.0f, 0.04f, 0.5f, -0.4f,
                                0.7f, -0.05f, 0.15f});
    const std::vector<std::string> names{
        "asset_0.spot", "asset_1.spot", "asset_1.variance"
    };
    constexpr double cross = 0.25;
    constexpr double rho = -0.4;
    const double covariance = cross * rho;
    const std::vector<double> dense{
        1.0, cross, covariance,
        cross, 1.0, rho,
        covariance, rho, 1.0
    };
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"jump", names, wb::correlation::packed_cholesky(dense, 3U)}
    };
    const std::vector<Product::ProductParameters> products{{0.95f, 21U}};
    std::vector<Expression> slots;
    for(std::size_t i=0U;i<Pack::kAssetCount+2U;++i)
        slots.push_back(Expression::source(i,1U));
    const auto plan=wb::compile_row_index_plan(
        Expression::aligned(slots),slots.size());
    const auto p = profile(4096U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, products, matrices, names, plan,
                           p, 1.0f / 252.0f, 1U);
    const auto cooperative = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, products, matrices, names, plan,
                           p, 1.0f / 252.0f, 1U);
    require(std::isfinite(scalar.prices[0]) && scalar.prices[0] > 0.0f
            && std::isfinite(cooperative.prices[0]),
            "CEV/Bates worst-of price is invalid.");
    require(scalar.prices[0] == cooperative.prices[0]
            && scalar.standard_errors[0] == cooperative.standard_errors[0],
            "Auxiliary QE/jump stream depends on path-team size.");
}

void check_jump_diffusions() {
    using Merton = wb::model::equity::merton::DynamicsPolicy;
    using Kou = wb::model::equity::kou::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<Merton, 1U>,
        wb::simulation::ModelGroup<Kou, 1U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                           0.8f, -0.05f, 0.15f});
    host.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                                0.8f, 0.4f, 5.0f, 6.0f});
    const std::vector<std::string> names{
        "asset_0.spot", "asset_1.spot"
    };
    const std::vector<double> dense{1.0, 0.3, 0.3, 1.0};
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"jumps", names, wb::correlation::packed_cholesky(dense, 2U)}
    };
    const std::vector<Product::ProductParameters> products{{0.95f, 21U}};
    std::vector<Expression> slots;
    for(std::size_t i=0U;i<Pack::kAssetCount+2U;++i)
        slots.push_back(Expression::source(i,1U));
    const auto plan=wb::compile_row_index_plan(
        Expression::aligned(slots),slots.size());
    const auto p = profile(4096U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, products, matrices, names, plan,
                           p, 1.0f / 252.0f, 1U);
    const auto cooperative = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, products, matrices, names, plan,
                           p, 1.0f / 252.0f, 1U);
    require(std::isfinite(scalar.prices[0]) && scalar.prices[0] > 0.0f,
            "Merton/Kou worst-of price is invalid.");
    require(scalar.prices == cooperative.prices
            && scalar.standard_errors == cooperative.standard_errors,
            "Merton/Kou auxiliary jumps depend on path-team size.");
}

void check_exact_ou_bridges() {
    using Stein = wb::model::equity::stein_stein::DynamicsPolicy;
    using Schobel = wb::model::equity::schobel_zhu::DynamicsPolicy;
    using Heston32 = wb::model::equity::heston_3_2::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<Stein, 1U>,
        wb::simulation::ModelGroup<Schobel, 1U>,
        wb::simulation::ModelGroup<Heston32, 1U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                           2.0f, 0.3f, -0.3f});
    host.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                                2.0f, 0.2f, 0.3f, -0.25f});
    host.tail.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.04f,
                                     2.0f, 0.04f, 0.3f, -0.5f});
    const std::vector<std::string> names{
        "asset_0.spot", "asset_0.volatility",
        "asset_1.spot", "asset_1.volatility",
        "asset_2.spot", "asset_2.volatility"
    };
    std::vector<double> dense(36U, 0.0);
    for (std::size_t i = 0; i < 6U; ++i) dense[i * 6U + i] = 1.0;
    for (std::size_t asset = 0; asset < 3U; ++asset) {
        const double rho = asset == 0U ? -0.3
            : asset == 1U ? -0.25 : -0.5;
        dense[(2U * asset) * 6U + 2U * asset + 1U] = rho;
        dense[(2U * asset + 1U) * 6U + 2U * asset] = rho;
    }
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"ou", names, wb::correlation::packed_cholesky(dense, 6U)}
    };
    const std::vector<Product::ProductParameters> products{{0.95f, 21U}};
    std::vector<Expression> slots;
    for(std::size_t i=0U;i<Pack::kAssetCount+2U;++i)
        slots.push_back(Expression::source(i,1U));
    const auto plan=wb::compile_row_index_plan(
        Expression::aligned(slots),slots.size());
    const auto p = profile(4096U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, products, matrices, names, plan,
                           p, 1.0f / 252.0f, 1U);
    const auto cooperative = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, products, matrices, names, plan,
                           p, 1.0f / 252.0f, 1U);
    require(std::isfinite(scalar.prices[0]) && scalar.prices[0] > 0.0f,
            "OU/Heston 3/2 worst-of price is invalid.");
    require(scalar.prices == cooperative.prices
            && scalar.standard_errors == cooperative.standard_errors,
            "OU bridge or Heston 3/2 depends on path-team size.");
}


void check_terminal_basket_products() {
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<BS, 1U>,
        wb::simulation::ModelGroup<BS, 1U>>;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models = {{1.0f, 0.0f, 0.2f, 0.0f}};
    host.tail.models = {{2.0f, 0.0f, 0.0f, 0.0f}};
    const std::vector<std::string> names{"first.spot", "second.spot"};
    const std::vector<wb::correlation::MatrixRow> matrices{{
        "identity", names, wb::correlation::packed_cholesky(
            {1.0, 0.0, 0.0, 1.0}, 2U)}};
    const auto plan = wb::compile_row_index_plan(
        Expression::aligned({
            Expression::source(0U, 1U), Expression::source(1U, 1U),
            Expression::source(2U, 1U), Expression::source(3U, 1U)
        }), 4U);
    const auto p = profile(256U);
    const float strike = 0.99f;
    const auto worst = wb::offline::pricing::execute_multi_asset_prices<
        Pack, wb::product::WorstOfCallPathPolicy, 4U>(
            host, {{strike, 21U}}, matrices, names, plan,
            p, 1.0f / 252.0f, 1U);
    const auto best = wb::offline::pricing::execute_multi_asset_prices<
        Pack, wb::product::BestOfCallPathPolicy, 4U>(
            host, {{strike, 21U}}, matrices, names, plan,
            p, 1.0f / 252.0f, 1U);
    const auto basket = wb::offline::pricing::execute_multi_asset_prices<
        Pack, wb::product::BasketCallPathPolicy, 4U>(
            host, {{strike, 21U}}, matrices, names, plan,
            p, 1.0f / 252.0f, 1U);
    const float first_ratio = std::exp(-0.2f * 21.0f / 252.0f);
    require(std::abs(worst.prices[0]
            - std::max(first_ratio - strike, 0.0f)) < 2e-6f,
            "worst-of basket reduction is incorrect");
    require(std::abs(best.prices[0] - (1.0f - strike)) < 2e-6f,
            "best-of basket reduction is incorrect");
    require(std::abs(basket.prices[0]
            - (0.5f * (first_ratio + 1.0f) - strike)) < 2e-6f,
            "arithmetic-mean basket reduction is incorrect");
}



int main() {
    check_cholesky();
    check_thirty_assets();
    check_stochastic_volatility();
    check_auxiliary_jump_stream();
    check_jump_diffusions();
    check_exact_ou_bridges();
    check_terminal_basket_products();
}
