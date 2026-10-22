// Gaussian factor coupling for exact NIG and VG one-step transitions.
#include "common/check_cuda.cuh"
#include "correlation/dataset.hpp"
#include "model/equity/markovian/normal_inverse_gaussian/dynamics_impl.cuh"
#include "model/equity/markovian/variance_gamma/dynamics_impl.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/multi_asset_price_generation.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ai_factory::workbench;
using VG = model::equity::variance_gamma::DynamicsPolicy;
using NIG = model::equity::normal_inverse_gaussian::DynamicsPolicy;
using Pack = simulation::MultiAssetPack<
    simulation::ModelGroup<VG,1U>, simulation::ModelGroup<NIG,1U>>;
using Product = product::WorstOfCallPathPolicy;
constexpr std::size_t paths = 8192U;
constexpr std::uint32_t steps = 16U;
constexpr std::uint64_t seed = 104244U;
constexpr float dt = 1.0f/252.0f;

__global__ void separate_reference(
    const VG::Parameters* vg_models, const NIG::Parameters* nig_models,
    const float* lower, float* payoffs
) {
    const auto path = blockIdx.x*blockDim.x+threadIdx.x;
    if (path>=paths) return;
    const auto key=philox::make_key(seed);
    const auto vg=VG::prepare_dynamics(vg_models[0],dt);
    const auto nig=NIG::prepare_dynamics(nig_models[0],dt);
    auto vg_state=VG::initial_state(vg);
    auto nig_state=NIG::initial_state(nig);
    for (std::uint32_t step=0U;step<steps;++step) {
        simulation::BrownianNormals<2U,1U> independent;
        independent.generate(key,path,step,0U);
        simulation::CorrelatedBrownianStep<2U,1U> correlated;
        correlated.calculate(independent,lower,0U);
        philox::DomainUniformSequence vg_aux(key,path,(2U<<24U)|step);
        VG::one_step_transition(vg,{correlated.get(0U,0U)},
                                vg_aux,vg_state);
    }
    for (std::uint32_t step=0U;step<steps;++step) {
        simulation::BrownianNormals<2U,1U> independent;
        independent.generate(key,path,step,0U);
        simulation::CorrelatedBrownianStep<2U,1U> correlated;
        correlated.calculate(independent,lower,0U);
        philox::DomainUniformSequence nig_aux(key,path,(3U<<24U)|step);
        NIG::one_step_transition(nig,{correlated.get(1U,0U)},
                                 nig_aux,nig_state);
    }
    const float minimum=fminf(VG::spot(vg_state)/vg_models[0].spot,
                              NIG::spot(nig_state)/nig_models[0].spot);
    payoffs[path]=fmaxf(minimum-0.9f,0.0f);
}

void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
}

int main() {
    const VG::Parameters vg{1.0f,0.0f,0.0f,0.2f,0.15f,-0.05f};
    const NIG::Parameters nig{1.0f,0.0f,0.0f,10.0f,-2.0f,0.2f};
    offline::pricing::MultiAssetHostModels<Pack> host;
    host.models={vg};
    host.tail.models={nig};
    const std::vector<std::string> factors{"vg.normal","nig.normal"};
    const std::vector<correlation::MatrixRow> matrices{{
        "coupled_levy",factors,
        correlation::packed_cholesky({1.0,0.35,0.35,1.0},2U)
    }};
    const std::vector<Product::ProductParameters> products{{0.9f,steps}};
    using Expression=RowIndexExpression;
    const auto plan=compile_row_index_plan(Expression::aligned({
        Expression::source(0U,1U),Expression::source(1U,1U),
        Expression::source(2U,1U),Expression::source(3U,1U)}),4U);
    offline::pricing::BatchedMonteCarloProfile profile{};
    profile.paths_per_price=paths;
    profile.results_per_launch=1U;
    profile.block_count_limit=1U;
    profile.threads_per_block=128U;
    profile.seed=seed;
    const auto run1=offline::pricing::execute_multi_asset_prices<
        Pack,Product,1U>(host,products,matrices,factors,plan,profile,dt,1U);
    const auto run4=offline::pricing::execute_multi_asset_prices<
        Pack,Product,4U>(host,products,matrices,factors,plan,profile,dt,1U);
    require(run1.prices==run4.prices
            && run1.standard_errors==run4.standard_errors,
            "NIG/VG price changed with path team size");
    offline::cuda::DeviceBuffer<VG::Parameters> d_vg(1U);
    offline::cuda::DeviceBuffer<NIG::Parameters> d_nig(1U);
    offline::cuda::DeviceBuffer<float> d_lower(3U), d_payoffs(paths);
    d_vg.copy_from(&vg);
    d_nig.copy_from(&nig);
    d_lower.copy_from(matrices[0].lower.data());
    separate_reference<<<paths/128U,128U>>>(
        d_vg.data(),d_nig.data(),d_lower.data(),d_payoffs.data());
    check_cuda(cudaGetLastError(),"NIG/VG reference launch");
    std::vector<float> reference(paths);
    d_payoffs.copy_to(reference.data());
    double mean=0.0;
    for(float payoff:reference) {
        require(std::isfinite(payoff),"NIG/VG payoff is not finite");
        mean+=payoff;
    }
    mean/=static_cast<double>(paths);
    require(std::abs(mean-run1.prices[0])<2.0e-6,
            "NIG/VG joint price differs from sequential marginals");

}
