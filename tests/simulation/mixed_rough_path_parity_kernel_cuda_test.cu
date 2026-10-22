// Joint rough/Markov payoff versus the same correlated marginal paths.
#include "cuda/check_cuda.cuh"
#include "multi_asset/rough_price_kernel.cuh"
#include "volterra/causal_fft_volatility_kernel.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/cuda/pricing_runner.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace wb=ai_factory::workbench;
namespace causal=wb::volterra::causal_fft;
using BS=wb::model::equity::black_scholes::DynamicsPolicy;
using Rough=wb::model::equity::rough_heston::CausalFftPathPolicy;
using MarkovPack=wb::simulation::MultiAssetPack<
    wb::simulation::ModelGroup<BS,1U>>;
using RoughPack=wb::simulation::MultiRoughPack<Rough>;
using Product=wb::product::WorstOfCallPathPolicy;
constexpr unsigned steps=128U;
constexpr unsigned paths=512U;
constexpr std::uint64_t seed=8121009U;
constexpr float dt=1.0f/252.0f;

__global__ void separate_marginals_reference(
    const BS::Parameters* bs_models,const Rough::Parameters* rough_models,
    const float* lower,const float* volatility,float* payoffs
) {
    const auto path=blockIdx.x*blockDim.x+threadIdx.x;
    if(path>=paths) return;
    const auto key=wb::philox::make_key(seed);
    const auto bs=BS::prepare_dynamics(bs_models[0],dt);
    auto bs_state=BS::initial_state(bs);
    for(unsigned step=0U;step<steps;++step) {
        wb::simulation::BrownianNormals<3U,1U> independent;
        independent.generate(key,path,step,0U);
        wb::simulation::CorrelatedBrownianStep<3U,1U> brownian;
        brownian.calculate(independent,lower,0U);
        wb::philox::DomainUniformSequence auxiliary(
            key,path,(2U<<24U)|step);
        BS::one_step_transition(bs,{brownian.get(1U,0U)},
                                auxiliary,bs_state);
    }
    const auto rough=Rough::prepare_model(rough_models[0],dt);
    float rough_log_spot=rough.initial_log_spot;
    for(unsigned step=0U;step<steps;++step) {
        wb::simulation::BrownianNormals<3U,1U> independent;
        independent.generate(key,path,step,0U);
        wb::simulation::CorrelatedBrownianStep<3U,1U> brownian;
        brownian.calculate(independent,lower,0U);
        Rough::advance_spot_cell(rough,
            volatility[causal::history_index(path,step,steps)],
            brownian.get(2U,0U),rough_log_spot);
    }
    const float minimum=fminf(
        BS::spot(bs_state)/bs_models[0].spot,
        expf(rough_log_spot-rough.initial_log_spot));
    payoffs[path]=fmaxf(minimum-0.9f,0.0f);
}

int main() {
    const BS::Parameters bs{1.0f,0.0f,0.0f,0.2f};
    const Rough::Parameters rough{
        1.0f,0.0f,0.0f,0.04f,2.0f,0.08f,0.3f,0.12f,-0.4f};
    const Product::ProductParameters product{0.9f,steps};
    const float lower[6]{
        1.0f,0.2f,0.9797959f,-0.4f,-0.0204124f,0.9162873f};
    const wb::RowIndexSelector selectors[4]{
        {1U,1U,wb::RowIndexMode::Strided},
        {1U,1U,wb::RowIndexMode::Strided},
        {1U,1U,wb::RowIndexMode::Strided},
        {1U,1U,wb::RowIndexMode::Strided}};
    wb::offline::cuda::DeviceBuffer<BS::Parameters> d_bs(1U);
    wb::offline::cuda::DeviceBuffer<Rough::Parameters> d_rough(1U);
    wb::offline::cuda::DeviceBuffer<Product::ProductParameters> d_product(1U);
    wb::offline::cuda::DeviceBuffer<wb::RowIndexSelector> d_selectors(4U);
    wb::offline::cuda::DeviceBuffer<float> d_lower(6U);
    wb::offline::cuda::DeviceBuffer<float> d_volatility(paths*steps);
    wb::offline::cuda::DeviceBuffer<float> d_joint(paths),d_reference(paths);
    wb::offline::cuda::DeviceBuffer<wb::reductions::MomentSums>
        d_moments(paths/128U);
    const auto plan=causal::plan_volatility_workspace<Rough>(steps,paths);
    wb::offline::cuda::DeviceBuffer<std::uint8_t> workspace(plan.bytes);
    d_bs.copy_from(&bs);
    d_rough.copy_from(&rough);
    d_product.copy_from(&product);
    d_selectors.copy_from(selectors);
    d_lower.copy_from(lower);
    causal::launch_volatility_factor_cuda<Rough,3U>(
        d_rough.data(),d_lower.data(),0U,0U,paths,dt,steps,32U,seed,
        workspace.data(),plan.bytes,d_volatility.data(),0U,3U,0,
        d_selectors.data(),1U,3U);
    const wb::monte_carlo::HeterogeneousRoughDeviceInputs<
        MarkovPack,RoughPack,Product> inputs{
            {{d_bs.data(),d_selectors.data()},{}},
            {d_rough.data(),{},d_selectors.data()+1U},
            d_product.data(),d_lower.data(),d_selectors.data()};
    wb::monte_carlo::launch_heterogeneous_rough_cuda<
        MarkovPack,RoughPack,Product,true,1U>(
            inputs,0U,0U,paths,steps,1U,dt,seed,
            d_volatility.data(),d_moments.data(),d_joint.data());
    separate_marginals_reference<<<paths/128U,128U>>>(
        d_bs.data(),d_rough.data(),d_lower.data(),
        d_volatility.data(),d_reference.data());
    wb::check_cuda(cudaGetLastError(),"separate rough marginal launch");
    wb::check_cuda(cudaDeviceSynchronize(),"rough path parity synchronize");
    std::vector<float> joint(paths),reference(paths);
    d_joint.copy_to(joint.data());
    d_reference.copy_to(reference.data());
    float maximum=0.0f;
    for(unsigned path=0U;path<paths;++path) {
        if(!std::isfinite(joint[path]))
            throw std::runtime_error("joint rough payoff is nonfinite");
        maximum=std::max(maximum,std::abs(joint[path]-reference[path]));
    }
    if(maximum>2e-6f)
        throw std::runtime_error(
            "joint rough paths differ from separately simulated marginals");
}
