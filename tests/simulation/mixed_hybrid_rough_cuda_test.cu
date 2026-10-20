// Correlated Gaussian-Volterra FFT for every hybrid rough equity policy.
#include "common/check_cuda.cuh"
#include "correlation/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/hybrid_multi_asset.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/pricing/mixed_rough_price_generation.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace ai_factory::workbench;
using BS=model::equity::black_scholes::DynamicsPolicy;
using MarkovPack=simulation::MultiAssetPack<simulation::ModelGroup<BS,1U>>;
using Product=product::WorstOfCallPathPolicy;
namespace hybrid=volterra::hybrid_fft;
namespace causal=volterra::causal_fft;
constexpr std::uint32_t steps=64U;
constexpr std::size_t paths=512U;
constexpr float dt=1.0f/252.0f;
constexpr std::uint64_t seed=918313U;

void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}

template<typename Policy>
__global__ void direct_hybrid_reference(
    const BS::Parameters* bs_parameters,
    const typename Policy::Parameters* rough_parameters,
    const float* lower,
    float* volatility_paths,float* payoffs
) {
    const auto path=blockIdx.x*blockDim.x+threadIdx.x;
    if(path>=paths) return;
    const auto key=philox::make_key(seed);
    const auto bs=BS::prepare_dynamics(bs_parameters[0],dt);
    auto bs_state=BS::initial_state(bs);
    const auto model=Policy::prepare_model(rough_parameters[0],dt);
    const auto kernel=Policy::KernelPolicy::prepare(
        Policy::kernel_parameters(rough_parameters[0]),dt);
    auto rough_state=Policy::Path::initial_state(model);
    for(std::uint32_t step=0U;step<steps;++step) {
        const auto spot_normal=simulation::correlated_brownian_factor(
            key,path,step,2U,lower);
        const auto rough_normal=simulation::correlated_brownian_factor(
            key,path,step,0U,lower);
        float far=0.0f;
        for(std::uint32_t prior=0U;prior<step;++prior) {
            const float past=simulation::correlated_brownian_factor(
                key,path,prior,0U,lower);
            far+=sqrtf(dt)*past*Policy::KernelPolicy::far_cell_weight(
                kernel,step-prior+1U);
        }
        const auto quad=philox::domain_uniform_quad(
            key,path,0U,(3U<<24U)|step);
        const float singular=philox::box_muller(
            quad.first,quad.second).first;
        const float value=Policy::KernelPolicy::reconstruct_volterra_value(
            kernel,far,rough_normal,singular);
        const float variance=Policy::kUsesVolterraVariance
            ? Policy::KernelPolicy::volterra_variance(
                kernel,(step+1U)*dt):0.0f;
        const float current_volatility=[&] {
            if constexpr (requires { rough_state.variance; })
                return rough_state.variance;
            else return rough_state.volatility;
        }();
        volatility_paths[causal::history_index(path,step,steps)]=
            current_volatility;
        const float independent_spot=(spot_normal-model.rho*rough_normal)
            /model.orthogonal_correlation;
        Policy::Path::advance(model,value,variance,rough_normal,
                              independent_spot,rough_state);
        philox::DomainUniformSequence auxiliary(
            key,path,(2U<<24U)|step);
        BS::one_step_transition(bs,{simulation::correlated_brownian_factor(
            key,path,step,1U,lower)},auxiliary,bs_state);
    }
    const float minimum=fminf(BS::spot(bs_state)/bs_parameters[0].spot,
        Policy::Path::spot(rough_state)/rough_parameters[0].spot);
    payoffs[path]=fmaxf(minimum-0.9f,0.0f);
}

template<typename Policy>
void check_model(const typename Policy::Parameters& rough,
                 const char* model_name) {
    const BS::Parameters bs{1.0f,0.0f,0.0f,0.2f};
    const Product::ProductParameters product{0.9f,steps};
    const monte_carlo::MultiAssetPriceRow row{0U,0U,0U,0U,1.0f};
    const std::uint32_t index=0U;
    const std::vector<std::string> external{
        "bs.spot","rough.spot","rough.vol"};
    const std::vector<std::string> internal{
        "rough.vol","bs.spot","rough.spot"};
    const correlation::MatrixRow matrix{
        "hybrid",external,correlation::packed_cholesky({
            1.0,0.2,0.1,
            0.2,1.0,-0.4,
            0.1,-0.4,1.0
        },3U)};
    const auto ordered=correlation::reorder_factors(matrix,internal);
    offline::cuda::DeviceBuffer<BS::Parameters> d_bs(1U);
    offline::cuda::DeviceBuffer<typename Policy::Parameters> d_rough(1U);
    offline::cuda::DeviceBuffer<std::uint32_t> d_indices(1U);
    offline::cuda::DeviceBuffer<Product::ProductParameters> d_product(1U);
    offline::cuda::DeviceBuffer<monte_carlo::MultiAssetPriceRow> d_row(1U);
    offline::cuda::DeviceBuffer<float> d_lower(ordered.lower.size());
    offline::cuda::DeviceBuffer<float> d_variance(paths*steps);
    offline::cuda::DeviceBuffer<float> d_direct_variance(paths*steps);
    offline::cuda::DeviceBuffer<float> d_joint(paths);
    offline::cuda::DeviceBuffer<float> d_reference(paths);
    offline::cuda::DeviceBuffer<volterra::PartialMoments> moments(paths/128U);
    const auto plan=hybrid::plan_correlated_volatility_workspace<Policy>(
        steps,paths);
    offline::cuda::DeviceBuffer<std::uint8_t> workspace(plan.bytes);
    d_bs.copy_from(&bs);
    d_rough.copy_from(&rough);
    d_indices.copy_from(&index);
    d_product.copy_from(&product);
    d_row.copy_from(&row);
    d_lower.copy_from(ordered.lower.data());
    hybrid::launch_correlated_hybrid_volatility_cuda<Policy,3U>(
        d_rough.data(),d_indices.data(),d_row.data(),d_lower.data(),
        0U,0U,paths,dt,steps,seed,workspace.data(),workspace.size(),
        d_variance.data(),0U,3U);
    direct_hybrid_reference<Policy><<<paths/128U,128U>>>(
        d_bs.data(),d_rough.data(),d_lower.data(),
        d_direct_variance.data(),d_reference.data());
    check_cuda(cudaGetLastError(),"hybrid direct reference launch");
    monte_carlo::MixedAssetDeviceInputs<MarkovPack,Policy,Product> inputs{
        {{d_bs.data(),d_indices.data()},{}},d_rough.data(),
        d_indices.data(),d_product.data(),d_row.data(),d_lower.data()};
    monte_carlo::launch_joint_rough_cuda<MarkovPack,Policy,Product,true>(
        inputs,0U,0U,paths,steps,1U,dt,seed,
        d_variance.data(),moments.data(),d_joint.data());
    check_cuda(cudaDeviceSynchronize(),"hybrid mixed synchronize");
    std::vector<float> variance(paths*steps),direct(paths*steps);
    std::vector<float> joint(paths),reference(paths);
    d_variance.copy_to(variance.data());
    d_direct_variance.copy_to(direct.data());
    d_joint.copy_to(joint.data());
    d_reference.copy_to(reference.data());
    float max_variance_error=0.0f;
    float max_payoff_error=0.0f;
    for(std::size_t i=0U;i<paths*steps;++i) {
        require(std::isfinite(variance[i]),"hybrid variance is not finite");
        max_variance_error=std::max(max_variance_error,
            std::abs(variance[i]-direct[i]));
    }
    for(std::size_t i=0U;i<paths;++i) {
        require(std::isfinite(joint[i]),"hybrid payoff is not finite");
        max_payoff_error=std::max(max_payoff_error,
            std::abs(joint[i]-reference[i]));
    }
    if(max_variance_error>2.0e-3f || max_payoff_error>2.0e-3f)
        throw std::runtime_error(std::string(model_name)
            +" differs from direct Volterra reference: variance="
            +std::to_string(max_variance_error)
            +" payoff="+std::to_string(max_payoff_error));
    offline::pricing::MultiAssetHostModels<MarkovPack> host;
    host.models={bs};
    host.scenario_indices={0U};
    const auto run=offline::pricing::execute_mixed_rough_prices<
        MarkovPack,Policy,Product>(
            host,{rough},{0U},1U,{product},{matrix},external,internal,
            {row},paths,paths,dt,1U,32U,seed);
    double mean=0.0;
    for(float value:reference) mean+=value;
    mean/=paths;
    require(std::abs(run.prices[0]-mean)<2.0e-3,
        "hybrid host price differs from direct reference");
    if constexpr (std::is_same_v<Policy,
        model::equity::rough_multi_asset::HybridPolicy<
            model::equity::rough_bergomi::PathPolicy>>) {
        using Barrier=product::BasketDownAndOutCallPathPolicy;
        const auto basket=offline::pricing::execute_mixed_rough_prices<
            MarkovPack,Policy,Barrier>(
                host,{rough},{0U},1U,{{0.9f,0.8f,steps,4U}},
                {matrix},external,internal,{row},paths,paths,dt,1U,
                32U,seed);
        require(std::isfinite(basket.prices[0]),
            "hybrid rough basket barrier price is invalid");
    }
}
}

int main() {
    using namespace ai_factory::workbench::model::equity;
    check_model<rough_multi_asset::HybridPolicy<rough_bergomi::PathPolicy>>(
        {1.0f,0.0f,0.0f,0.04f,1.0f,0.12f,-0.4f},"rough Bergomi");
    check_model<rough_multi_asset::HybridPolicy<
        log_modulated_rough_bergomi::PathPolicy>>(
        {1.0f,0.0f,0.0f,0.04f,0.8f,0.12f,-0.4f,0.5f,1.0f},
        "log-modulated rough Bergomi");
    check_model<rough_multi_asset::HybridPolicy<rough_sabr::PathPolicy>>(
        {1.0f,0.0f,0.0f,0.2f,0.7f,0.12f,-0.4f,0.8f},"rough SABR");
    check_model<rough_multi_asset::HybridPolicy<rough_stein_stein::PathPolicy>>(
        {1.0f,0.0f,0.0f,0.2f,1.0f,0.15f,0.12f,-0.4f},
        "rough Stein-Stein");

    // Two rough variance factors use the 2 x 2 Cholesky prefix; the joint
    // spot scan then consumes the full 5 x 5 factorization.
    using Policy=rough_multi_asset::HybridPolicy<rough_bergomi::PathPolicy>;
    offline::pricing::MultiAssetHostModels<MarkovPack> host;
    host.models={{1.0f,0.0f,0.0f,0.2f}};
    host.scenario_indices={0U};
    const std::vector<Policy::Parameters> rough_models{
        {1.0f,0.0f,0.0f,0.04f,1.0f,0.12f,-0.4f},
        {1.0f,0.0f,0.0f,0.06f,0.8f,0.15f,-0.4f}};
    const std::vector<std::string> internal{
        "rough_0.vol","rough_1.vol","bs.spot",
        "rough_0.spot","rough_1.spot"};
    const std::vector<std::string> external{
        "bs.spot","rough_0.spot","rough_0.vol",
        "rough_1.spot","rough_1.vol"};
    const correlation::MatrixRow internal_matrix{
        "two_rough",internal,correlation::packed_cholesky({
            1.0,0.2,0.0,-0.4,0.0,
            0.2,1.0,0.0,0.0,-0.4,
            0.0,0.0,1.0,0.0,0.0,
            -0.4,0.0,0.0,1.0,0.0,
            0.0,-0.4,0.0,0.0,1.0
        },5U)};
    const auto external_matrix=correlation::reorder_factors(
        internal_matrix,external);
    const std::vector<monte_carlo::MultiAssetPriceRow> rows{
        {0U,0U,0U,0U,1.0f}};
    const auto price=offline::pricing::execute_mixed_rough_prices<
        MarkovPack,Policy,Product,2U>(
            host,rough_models,{0U,1U},1U,{{0.9f,steps}},
            {external_matrix},external,internal,rows,paths,paths,dt,1U,
            32U,seed);
    require(std::isfinite(price.prices[0]) && price.prices[0]>=0.0f,
        "two hybrid rough price is invalid");
    const std::vector<std::string> single_external{
        "bs.spot","rough.spot","rough.vol"};
    const std::vector<std::string> single_internal{
        "rough.vol","bs.spot","rough.spot"};
    const correlation::MatrixRow single_matrix{
        "single",single_external,correlation::packed_cholesky({
            1.0,0.0,0.0,0.0,1.0,-0.4,0.0,-0.4,1.0
        },3U)};
    const auto run_joint=[&] {
        return offline::pricing::execute_mixed_rough_prices<
            MarkovPack,Policy,Product,2U>(
                host,rough_models,{0U,1U},1U,{{0.9f,steps}},
                {external_matrix},external,internal,rows,
                paths,paths,dt,1U,32U,seed);
    };
    const auto run_single=[&](const Policy::Parameters& rough) {
        return offline::pricing::execute_mixed_rough_prices<
            MarkovPack,Policy,Product>(
                host,{rough},{0U},1U,{{0.9f,steps}},
                {single_matrix},single_external,single_internal,rows,
                paths,paths,dt,1U,32U,seed);
    };
    // The first launch of a new CUDA template can include module loading.
    run_joint();
    for(const auto& rough:rough_models) run_single(rough);
    std::vector<double> joint_times,sequential_times;
    for(unsigned trial=0U;trial<7U;++trial) {
        joint_times.push_back(run_joint().kernel_seconds);
        double separate=0.0;
        for(const auto& rough:rough_models)
            separate+=run_single(rough).kernel_seconds;
        sequential_times.push_back(separate);
    }
    std::sort(joint_times.begin(),joint_times.end());
    std::sort(sequential_times.begin(),sequential_times.end());
    std::cout<<"two_hybrid_joint_ms="<<joint_times[3U]*1000.0
             <<" two_hybrid_sequential_ms="
             <<sequential_times[3U]*1000.0<<'\n';
}
