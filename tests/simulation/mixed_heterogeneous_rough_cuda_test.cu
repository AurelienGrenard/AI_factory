#include "correlation/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/hybrid_multi_asset.cuh"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/heterogeneous_rough_price_generation.cuh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace wb=ai_factory::workbench;
using BS=wb::model::equity::black_scholes::DynamicsPolicy;
using Markov=wb::simulation::MultiAssetPack<
    wb::simulation::ModelGroup<BS,1U>>;
using Heston=wb::model::equity::rough_heston::CausalFftPathPolicy;
using Quadratic=wb::model::equity::quadratic_rough_heston::CausalFftPathPolicy;
using Bergomi=wb::model::equity::rough_multi_asset::HybridPolicy<
    wb::model::equity::rough_bergomi::PathPolicy>;
using Rough=wb::simulation::MultiRoughPack<Heston,Bergomi>;
using Worst=wb::product::WorstOfCallPathPolicy;
using Barrier=wb::product::BasketDownAndOutCallPathPolicy;
using Expression=wb::RowIndexExpression;

void require(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(message);
}

void check_one_factor_rough() {
    using Pack=wb::simulation::MultiRoughPack<Quadratic>;
    wb::offline::pricing::MultiAssetHostModels<Markov> markov;
    markov.models={{1.0f,0.0f,0.0f,0.2f}};
    wb::offline::pricing::MultiRoughHostModels<Pack> rough;
    rough.models={{1.0f,0.0f,0.0f,0.05f,0.7f,0.1f,
        0.02f,1.4f,0.3f,0.12f}};
    const std::vector<std::string> external{"bs.spot","quadratic.spot"};
    const std::vector<std::string> internal{"quadratic.spot","bs.spot"};
    const std::vector<wb::correlation::MatrixRow> matrices{{
        "one_factor",external,wb::correlation::packed_cholesky({
            1.0,0.2,0.2,1.0},2U)}};
    const auto plan=wb::compile_row_index_plan(Expression::aligned({
        Expression::source(0U,1U),Expression::source(1U,1U),
        Expression::source(2U,1U),Expression::source(3U,1U)}),4U);
    const auto one=wb::offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Pack,Worst,1U>(markov,rough,{{0.9f,128U}},matrices,
            external,internal,plan,256U,128U,1.0f/252.0f,1U,32U,8121009U);
    const auto four=wb::offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Pack,Worst,4U>(markov,rough,{{0.9f,128U}},matrices,
            external,internal,plan,256U,128U,1.0f/252.0f,1U,32U,8121009U);
    require(std::isfinite(one.prices[0]) && one.prices[0]>=0.0f,
        "one-factor quadratic rough price invalid");
    require(std::abs(one.prices[0]-four.prices[0])<2e-6f,
        "one-factor rough price depends on team size");
}

void check_large_mixed_basket() {
    using LargeMarkov=wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<BS,30U>>;
    using OneHeston=wb::simulation::MultiRoughPack<Heston>;
    wb::offline::pricing::MultiAssetHostModels<LargeMarkov> markov;
    markov.models={{1.0f,0.0f,0.0f,0.2f}};
    wb::offline::pricing::MultiRoughHostModels<OneHeston> rough;
    rough.models={{1.0f,0.0f,0.0f,0.04f,2.0f,0.08f,
        0.3f,0.12f,-0.4f}};
    std::vector<std::string> external;
    for(unsigned asset=0U;asset<30U;++asset)
        external.push_back("bs_"+std::to_string(asset)+".spot");
    external.push_back("rough.spot");
    external.push_back("rough.vol");
    std::vector<std::string> internal{"rough.vol"};
    internal.insert(internal.end(),external.begin(),external.begin()+30U);
    internal.push_back("rough.spot");
    std::vector<double> dense(32U*32U,0.0);
    for(unsigned factor=0U;factor<32U;++factor)
        dense[factor*32U+factor]=1.0;
    dense[30U*32U+31U]=dense[31U*32U+30U]=-0.4;
    const std::vector<wb::correlation::MatrixRow> matrices{{
        "large",external,wb::correlation::packed_cholesky(dense,32U)}};
    std::vector<Expression> slots;
    for(unsigned slot=0U;slot<33U;++slot)
        slots.push_back(Expression::source(slot,1U));
    const auto plan=wb::compile_row_index_plan(Expression::aligned(slots),33U);
    const auto one=wb::offline::pricing::execute_heterogeneous_rough_prices<
        LargeMarkov,OneHeston,Worst,1U>(markov,rough,{{0.0f,64U}},
            matrices,external,internal,plan,256U,128U,
            1.0f/252.0f,1U,32U,7331U);
    const auto four=wb::offline::pricing::execute_heterogeneous_rough_prices<
        LargeMarkov,OneHeston,Worst,4U>(markov,rough,{{0.0f,64U}},
            matrices,external,internal,plan,256U,128U,
            1.0f/252.0f,1U,32U,7331U);
    require(std::isfinite(one.prices[0]) && one.prices[0]>0.0f,
        "large rough/Markov basket price invalid");
    require(std::abs(one.prices[0]-four.prices[0])<2e-6f,
        "large rough/Markov basket depends on team size");
}

int main() {
    check_large_mixed_basket();
    check_one_factor_rough();
    constexpr std::size_t paths=512U;
    constexpr std::uint32_t steps=64U;
    constexpr float dt=1.0f/252.0f;
    constexpr std::uint64_t seed=919133U;
    wb::offline::pricing::MultiAssetHostModels<Markov> markov;
    markov.models={{1.0f,0.0f,0.0f,0.2f}};
    wb::offline::pricing::MultiRoughHostModels<Rough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.04f,2.0f,0.08f,
        0.3f,0.12f,-0.4f}};
    rough.tail.models={{1.0f,0.0f,0.0f,0.04f,1.0f,
        0.12f,-0.4f}};
    const std::vector<std::string> external{
        "bs.spot","heston.spot","heston.vol",
        "bergomi.spot","bergomi.vol"};
    const std::vector<std::string> internal{
        "heston.vol","bergomi.vol","bs.spot",
        "heston.spot","bergomi.spot"};
    const wb::correlation::MatrixRow internal_matrix{
        "heterogeneous",internal,wb::correlation::packed_cholesky({
            1.0,0.2,0.1,-0.4,0.0,
            0.2,1.0,0.0,0.0,-0.4,
            0.1,0.0,1.0,0.0,0.0,
            -0.4,0.0,0.0,1.0,0.0,
            0.0,-0.4,0.0,0.0,1.0},5U)};
    const auto matrix=wb::correlation::reorder_factors(
        internal_matrix,external);
    const auto plan=wb::compile_row_index_plan(Expression::aligned({
        Expression::source(0U,1U),Expression::source(1U,1U),
        Expression::source(2U,1U),Expression::source(3U,1U),
        Expression::source(4U,1U)}),5U);
    const std::vector<Worst::ProductParameters> products{{0.9f,steps}};
    const auto full=wb::offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Rough,Worst>(markov,rough,products,{matrix},
            external,internal,plan,paths,paths,dt,1U,32U,seed);
    const auto chunked=wb::offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Rough,Worst>(markov,rough,products,{matrix},
            external,internal,plan,paths,128U,dt,1U,32U,seed);
    require(std::isfinite(full.prices[0]) && full.prices[0]>=0.0f,
        "heterogeneous rough price invalid");
    require(full.prices==chunked.prices
        &&full.standard_errors==chunked.standard_errors,
        "heterogeneous rough price depends on chunking");
    const auto barrier=wb::offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Rough,Barrier,1U>(markov,rough,
            {{0.9f,0.8f,steps,4U}},{matrix},external,internal,
            plan,paths,paths,dt,1U,32U,seed);
    const auto barrier_team4=wb::offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Rough,Barrier,4U>(markov,rough,
            {{0.9f,0.8f,steps,4U}},{matrix},external,internal,
            plan,paths,paths,dt,1U,32U,seed);
    require(std::abs(barrier.prices[0]-barrier_team4.prices[0])<2e-6f,
        "rough basket barrier depends on team size");
    const auto batch_plan=wb::compile_row_index_plan(Expression::cartesian({
        Expression::aligned({Expression::source(0U,1U),
            Expression::source(1U,1U),Expression::source(2U,1U),
            Expression::source(4U,1U)}),
        Expression::source(3U,8U)}),5U);
    const std::vector<Worst::ProductParameters> batch_products(8U,{0.9f,steps});
    const auto serial=wb::offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Rough,Worst>(markov,rough,batch_products,{matrix},
            external,internal,batch_plan,paths,paths,dt,1U,32U,seed,1U);
    const auto parallel=wb::offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Rough,Worst>(markov,rough,batch_products,{matrix},
            external,internal,batch_plan,paths,paths,dt,1U,32U,seed,4U);
    require(serial.prices==parallel.prices
        &&serial.standard_errors==parallel.standard_errors,
        "rough price depends on stream batching");
}
