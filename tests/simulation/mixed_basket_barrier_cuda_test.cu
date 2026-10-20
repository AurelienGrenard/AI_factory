// Cross-asset basket barrier observed during the joint rough replay.
#include "common/check_cuda.cuh"
#include "common/monte_carlo/multi_asset_rough_kernel.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/pricing/mixed_rough_price_generation.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace causal = ai_factory::workbench::volterra::causal_fft;
using BS = model::equity::black_scholes::DynamicsPolicy;
using Rough = model::equity::rough_heston::CausalFftPathPolicy;
using Product = product::BasketDownAndOutCallPathPolicy;
constexpr std::uint32_t steps = 128U;
constexpr std::size_t paths = 4096U;
constexpr std::uint64_t seed = 92319U;
constexpr float dt = 1.0f / 252.0f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

__global__ void separate_barrier_reference(
    const BS::Parameters* bs_parameters,
    const Rough::Parameters* rough_parameters,
    const Product::ProductParameters* products,
    const float* lower, const float* volatility, float* payoffs
) {
    const auto path=blockIdx.x*blockDim.x+threadIdx.x;
    if (path>=paths) return;
    const auto key=philox::make_key(seed);
    const auto bs=BS::prepare_dynamics(bs_parameters[0],dt);
    const auto rough=Rough::prepare_model(rough_parameters[0],dt);
    auto bs_state=BS::initial_state(bs);
    float rough_log_spot=rough.initial_log_spot;
    bool breached=false;
    float basket=1.0f;
    for (std::uint32_t step=0U;step<steps;++step) {
        simulation::BrownianNormals<3U,1U> independent;
        independent.generate(key,path,step,0U);
        simulation::CorrelatedBrownianStep<3U,1U> correlated;
        correlated.calculate(independent,lower,0U);
        philox::DomainUniformSequence auxiliary(
            key,path,(2U<<24U)|step);
        BS::one_step_transition(bs,{correlated.get(1U,0U)},
                                auxiliary,bs_state);
        Rough::advance_spot_cell(
            rough,volatility[causal::history_index(path,step,steps)],
            correlated.get(2U,0U),rough_log_spot);
        const auto product=products[0];
        if (Product::is_observation_step(step,steps,1U,product)) {
            basket=0.5f*(BS::spot(bs_state)/bs_parameters[0].spot
                +expf(rough_log_spot-rough.initial_log_spot));
            breached |= basket<=product.lower_barrier;
        }
    }
    payoffs[path]=breached ? 0.0f : fmaxf(basket-products[0].strike,0.0f);
}

template<std::size_t MarkovCount>
struct Fixture {
    using Pack = simulation::MultiAssetPack<
        simulation::ModelGroup<BS, MarkovCount>>;
    static constexpr std::size_t dimension = MarkovCount + 2U;
    const BS::Parameters bs{1.0f, 0.0f, 0.0f, 0.2f};
    const Rough::Parameters rough{
        1.0f, 0.0f, 0.0f, 0.04f, 2.0f, 0.08f, 0.3f, 0.12f, -0.4f
    };
    const monte_carlo::MultiAssetPriceRow row{0U, 0U, 0U, 0U, 1.0f};
    const std::uint32_t index = 0U;
    std::vector<float> lower;
    std::vector<std::uint32_t> indices;
    offline::cuda::DeviceBuffer<BS::Parameters> d_bs{1U};
    offline::cuda::DeviceBuffer<Rough::Parameters> d_rough{1U};
    offline::cuda::DeviceBuffer<std::uint32_t> d_indices{MarkovCount};
    offline::cuda::DeviceBuffer<std::uint32_t> d_rough_index{1U};
    offline::cuda::DeviceBuffer<Product::ProductParameters> d_product{1U};
    offline::cuda::DeviceBuffer<monte_carlo::MultiAssetPriceRow> d_row{1U};
    offline::cuda::DeviceBuffer<float> d_lower{dimension*(dimension+1U)/2U};
    offline::cuda::DeviceBuffer<float> d_volatility{paths*steps};
    offline::cuda::DeviceBuffer<float> d_joint{paths};
    offline::cuda::DeviceBuffer<float> d_reference{paths};
    offline::cuda::DeviceBuffer<volterra::PartialMoments> d_moments{paths/128U};
    causal::VolatilityWorkspacePlan volatility_plan =
        causal::plan_volatility_workspace<Rough>(steps, paths);
    offline::cuda::DeviceBuffer<std::uint8_t> d_workspace{
        volatility_plan.bytes};
    monte_carlo::MixedAssetDeviceInputs<Pack, Rough, Product> inputs;

    Fixture() : lower(dimension*(dimension+1U)/2U, 0.0f),
        indices(MarkovCount, 0U),
        inputs{{{d_bs.data(), d_indices.data()}, {}}, d_rough.data(),
               d_rough_index.data(), d_product.data(), d_row.data(),
               d_lower.data()} {
        for (std::size_t i=0U; i<dimension; ++i)
            lower[i*(i+1U)/2U+i]=1.0f;
        const auto spot=dimension-1U;
        lower[spot*(spot+1U)/2U]=-0.4f;
        lower[spot*(spot+1U)/2U+spot]=std::sqrt(0.84f);
        d_bs.copy_from(&bs);
        d_rough.copy_from(&rough);
        d_indices.copy_from(indices.data());
        d_rough_index.copy_from(&index);
        d_row.copy_from(&row);
        d_lower.copy_from(lower.data());
    }

    void set_product(std::uint32_t interval, float barrier) {
        Product::ProductParameters product{0.9f, barrier, steps, interval};
        d_product.copy_from(&product);
    }

    void volatility_first() {
        causal::launch_volatility_cuda<Rough, dimension, 0U, dimension>(
            d_rough.data(), d_rough_index.data(), d_row.data(),
            d_lower.data(), 0U, 0U, paths, dt, steps, 32U, seed,
            d_workspace.data(), d_workspace.size(), d_volatility.data()
        );
        monte_carlo::launch_joint_rough_cuda<Pack,Rough,Product,true>(
            inputs, 0U, 0U, paths, steps, 1U, dt, seed,
            d_volatility.data(), d_moments.data(), d_joint.data());
    }

    void compare() {
        volatility_first();
        check_cuda(cudaDeviceSynchronize(), "basket synchronize");
        std::vector<float> joint(paths);
        d_joint.copy_to(joint.data());
        for (float payoff : joint)
            require(std::isfinite(payoff), "basket payoff is not finite");
        if constexpr (MarkovCount == 1U) {
            separate_barrier_reference<<<paths/128U,128U>>>(
                d_bs.data(),d_rough.data(),d_product.data(),
                d_lower.data(),d_volatility.data(),d_reference.data());
            check_cuda(cudaGetLastError(),"basket separate reference launch");
            std::vector<float> reference(paths);
            d_reference.copy_to(reference.data());
            for (std::size_t i=0U;i<paths;++i)
                require(std::abs(joint[i]-reference[i])<2.0e-6f,
                    "joint basket payoff differs from separate marginals");
        }
    }

    float time_ms() {
        offline::cuda::Event start, stop;
        volatility_first();
        check_cuda(cudaDeviceSynchronize(), "basket timing warmup");
        check_cuda(cudaEventRecord(start.get()), "basket timing start");
        for (unsigned repeat=0U; repeat<12U; ++repeat) {
            volatility_first();
        }
        check_cuda(cudaEventRecord(stop.get()), "basket timing stop");
        check_cuda(cudaEventSynchronize(stop.get()), "basket timing wait");
        float ms=0.0f;
        check_cuda(cudaEventElapsedTime(&ms,start.get(),stop.get()),
            "basket timing elapsed");
        return ms/12.0f;
    }
};
}

int main() {
    {
        using PurePack=simulation::MultiAssetPack<
            simulation::ModelGroup<BS,2U>>;
        offline::pricing::MultiAssetHostModels<PurePack> host;
        host.models={{1.0f,0.0f,0.0f,0.0f}};
        host.scenario_indices={0U,0U};
        const std::vector<std::string> factors{"asset_0.spot","asset_1.spot"};
        const std::vector<correlation::MatrixRow> matrices{{
            "identity",factors,correlation::packed_cholesky(
                {1.0,0.0,0.0,1.0},2U)}};
        const std::vector<monte_carlo::MultiAssetPriceRow> rows{
            {0U,0U,0U,0U,1.0f}};
        offline::pricing::BatchedMonteCarloProfile profile{};
        profile.paths_per_price=256U;
        profile.results_per_launch=1U;
        profile.block_count_limit=1U;
        profile.threads_per_block=128U;
        profile.seed=seed;
        const auto price=[&](float barrier) {
            const std::vector<Product::ProductParameters> products{
                {0.9f,barrier,steps,4U}};
            return offline::pricing::execute_multi_asset_prices<
                PurePack,Product,4U>(
                    host,1U,products,matrices,factors,rows,profile,dt,1U);
        };
        require(std::abs(price(0.85f).prices[0]-0.1f)<2.0e-6f,
            "deterministic basket barrier call has wrong payoff");
        require(price(1.15f).prices[0]==0.0f,
            "deterministic basket should be knocked out");
        host.models[0].volatility=0.2f;
        const std::vector<Product::ProductParameters> stochastic_product{
            {0.9f,0.8f,steps,4U}};
        const auto single=offline::pricing::execute_multi_asset_prices<
            PurePack,Product,1U>(host,1U,stochastic_product,matrices,
                factors,rows,profile,dt,1U);
        const auto four=offline::pricing::execute_multi_asset_prices<
            PurePack,Product,4U>(host,1U,stochastic_product,matrices,
                factors,rows,profile,dt,1U);
        require(single.prices==four.prices
            && single.standard_errors==four.standard_errors,
            "basket barrier price depends on path team size");
    }
    Fixture<1U> small;
    for (const auto interval: {1U, 4U, 16U}) {
        small.set_product(interval, 0.85f);
        small.compare();
        std::cout << "basket_assets=2 interval=" << interval
                  << " volatility_first_ms=" << small.time_ms()
                  << '\n';
    }
    small.set_product(4U, 1.15f);
    small.compare();
    {
        using Pack=Fixture<1U>::Pack;
        offline::pricing::MultiAssetHostModels<Pack> host;
        host.models={small.bs};
        host.scenario_indices={0U};
        const std::vector<Rough::Parameters> rough_models{small.rough};
        const std::vector<std::uint32_t> rough_indices{0U};
        const std::vector<Product::ProductParameters> products{
            {0.9f,0.85f,steps,4U}};
        const std::vector<std::string> external{
            "bs.spot","rough.spot","rough.vol"};
        const std::vector<std::string> internal{
            "rough.vol","bs.spot","rough.spot"};
        const std::vector<correlation::MatrixRow> correlations{{
            "basket",external,correlation::packed_cholesky(
                {1.0,0.0,0.0, 0.0,1.0,-0.4, 0.0,-0.4,1.0},3U)
        }};
        const std::vector<monte_carlo::MultiAssetPriceRow> rows{
            {0U,0U,0U,0U,1.0f}};
        const auto run=[&](std::size_t chunk) {
            return offline::pricing::execute_mixed_rough_prices<
                Pack,Rough,Product>(
                    host,rough_models,rough_indices,1U,products,
                    correlations,external,internal,rows,paths,chunk,dt,1U,
                    32U,seed);
        };
        const auto full=run(paths);
        const auto chunked=run(512U);
        require(full.prices==chunked.prices
            && full.standard_errors==chunked.standard_errors,
            "basket price depends on path chunking");
    }
    Fixture<30U> large;
    large.set_product(4U, 0.85f);
    large.compare();
    std::cout << "basket_assets=31 interval=4"
              << " volatility_first_ms=" << large.time_ms()
              << '\n';
}
