// One Markovian plus two rough assets against separate marginals.
#include "common/check_cuda.cuh"
#include "common/monte_carlo/multi_asset_rough_kernel.cuh"
#include "correlation/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/pricing/mixed_rough_price_generation.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace causal = ai_factory::workbench::volterra::causal_fft;
using BS = model::equity::black_scholes::DynamicsPolicy;
using Rough = model::equity::rough_heston::CausalFftPathPolicy;
using Product = product::WorstOfCallPathPolicy;
using MarkovPack = simulation::MultiAssetPack<simulation::ModelGroup<BS, 1U>>;
constexpr unsigned paths = 512U;
constexpr unsigned steps = 64U;
constexpr std::uint64_t seed = 711U;

__global__ void separate_reference(
    const BS::Parameters* bs_parameters,
    const Rough::Parameters* rough_parameters,
    const float* lower,
    const float* volatility,
    float* payoffs
) {
    const unsigned path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    const auto key = philox::make_key(seed);
    const auto bs = BS::prepare_dynamics(bs_parameters[0], 1.0f / 252.0f);
    auto state = BS::initial_state(bs);
    for (unsigned step = 0U; step < steps; ++step) {
        simulation::BrownianNormals<5U, 1U> independent;
        independent.generate(key, path, step, 0U);
        simulation::CorrelatedBrownianStep<5U, 1U> brownian;
        brownian.calculate(independent, lower, 0U);
        philox::DomainUniformSequence auxiliary(
            key, path, (2U << 24U) | step
        );
        BS::one_step_transition(bs, {brownian.get(2U, 0U)},
            auxiliary, state);
    }
    float minimum = BS::spot(state) / bs_parameters[0].spot;
    for (unsigned asset = 0U; asset < 2U; ++asset) {
        const auto rough = Rough::prepare_model(
            rough_parameters[asset], 1.0f / 252.0f
        );
        float log_spot = rough.initial_log_spot;
        for (unsigned step = 0U; step < steps; ++step) {
            simulation::BrownianNormals<5U, 1U> independent;
            independent.generate(key, path, step, 0U);
            simulation::CorrelatedBrownianStep<5U, 1U> brownian;
            brownian.calculate(independent, lower, 0U);
            const float variance = volatility[
                asset * paths * steps + causal::history_index(path, step, steps)
            ];
            Rough::advance_spot_cell(rough, variance,
                brownian.get(3U + asset, 0U), log_spot);
        }
        minimum = fminf(minimum,
            expf(log_spot - rough.initial_log_spot));
    }
    payoffs[path] = fmaxf(minimum - 0.9f, 0.0f);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    const BS::Parameters bs{1.0f, 0.0f, 0.0f, 0.2f};
    const Rough::Parameters rough[2]{
        {1.0f, 0.0f, 0.0f, 0.04f, 2.0f, 0.08f, 0.3f, 0.12f, -0.4f},
        {1.0f, 0.0f, 0.0f, 0.06f, 1.6f, 0.10f, 0.25f, 0.18f, -0.3f}
    };
    const Product::ProductParameters product{0.9f, steps};
    const monte_carlo::MultiAssetPriceRow row{0U, 0U, 0U, 0U, 1.0f};
    const std::uint32_t indices[2]{0U, 1U};
    // Internal factor order: rough vol 0/1, BS spot, rough spot 0/1.
    const std::vector<double> matrix{
        1.0, 0.1, 0.2, -0.4, 0.0,
        0.1, 1.0, -0.05, 0.0, -0.3,
        0.2, -0.05, 1.0, 0.1, 0.05,
        -0.4, 0.0, 0.1, 1.0, 0.05,
        0.0, -0.3, 0.05, 0.05, 1.0
    };
    const auto lower = correlation::packed_cholesky(matrix, 5U);
    offline::cuda::DeviceBuffer<BS::Parameters> device_bs(1U);
    offline::cuda::DeviceBuffer<Rough::Parameters> device_rough(2U);
    offline::cuda::DeviceBuffer<std::uint32_t> device_bs_index(1U);
    offline::cuda::DeviceBuffer<std::uint32_t> device_rough_index(2U);
    offline::cuda::DeviceBuffer<Product::ProductParameters> device_product(1U);
    offline::cuda::DeviceBuffer<monte_carlo::MultiAssetPriceRow> device_row(1U);
    offline::cuda::DeviceBuffer<float> device_lower(lower.size());
    offline::cuda::DeviceBuffer<float> volatility(2U * paths * steps);
    offline::cuda::DeviceBuffer<float> joint_payoffs(paths);
    offline::cuda::DeviceBuffer<float> reference_payoffs(paths);
    offline::cuda::DeviceBuffer<volterra::PartialMoments> moments(paths / 128U);
    const auto plan = causal::plan_volatility_workspace<Rough>(steps, paths);
    offline::cuda::DeviceBuffer<std::uint8_t> workspace(plan.bytes);
    device_bs.copy_from(&bs);
    device_rough.copy_from(rough);
    device_bs_index.copy_from(indices);
    device_rough_index.copy_from(indices);
    device_product.copy_from(&product);
    device_row.copy_from(&row);
    device_lower.copy_from(lower.data());
    for (std::size_t asset = 0U; asset < 2U; ++asset) {
        causal::launch_volatility_factor_cuda<Rough, 5U>(
            device_rough.data(), device_rough_index.data(),
            device_row.data(), device_lower.data(), 0U, 0U,
            paths, 1.0f / 252.0f, steps, 32U, seed,
            workspace.data(), workspace.size(),
            volatility.data() + asset * paths * steps,
            asset, static_cast<std::uint32_t>(3U + asset), 2U, asset
        );
    }
    monte_carlo::MixedAssetDeviceInputs<MarkovPack, Rough, Product, 2U>
        inputs{{{device_bs.data(), device_bs_index.data()}, {}},
            device_rough.data(), device_rough_index.data(),
            device_product.data(), device_row.data(), device_lower.data()};
    monte_carlo::launch_joint_rough_cuda<MarkovPack, Rough, Product, true>(
        inputs, 0U, 0U, paths, steps, 1U, 1.0f / 252.0f, seed,
        volatility.data(), moments.data(), joint_payoffs.data()
    );
    separate_reference<<<paths / 128U, 128U>>>(
        device_bs.data(), device_rough.data(), device_lower.data(),
        volatility.data(), reference_payoffs.data()
    );
    check_cuda(cudaGetLastError(), "two rough reference launch");
    check_cuda(cudaDeviceSynchronize(), "two rough synchronize");
    std::vector<float> joint(paths), reference(paths);
    joint_payoffs.copy_to(joint.data());
    reference_payoffs.copy_to(reference.data());
    for (std::size_t path = 0U; path < paths; ++path) {
        require(std::isfinite(joint[path]), "two rough payoff is not finite");
        require(std::abs(joint[path] - reference[path]) < 2.0e-6f,
            "two rough joint path disagrees with separate marginals");
    }
    // The crossed barrier observes both rough spots in the joint scan.
    using Barrier = product::BasketDownAndOutCallPathPolicy;
    const Barrier::ProductParameters barrier{0.9f, 0.85f, steps, 4U};
    offline::cuda::DeviceBuffer<Barrier::ProductParameters> barrier_product(1U);
    barrier_product.copy_from(&barrier);
    monte_carlo::MixedAssetDeviceInputs<MarkovPack, Rough, Barrier, 2U>
        barrier_inputs{{{device_bs.data(), device_bs_index.data()}, {}},
            device_rough.data(), device_rough_index.data(),
            barrier_product.data(), device_row.data(), device_lower.data()};
    monte_carlo::launch_joint_rough_cuda<MarkovPack, Rough, Barrier, true>(
        barrier_inputs, 0U, 0U, paths, steps, 1U, 1.0f / 252.0f, seed,
        volatility.data(), moments.data(), joint_payoffs.data());
    joint_payoffs.copy_to(joint.data());
    for (float payoff : joint)
        require(std::isfinite(payoff), "two rough basket payoff is not finite");

    offline::pricing::MultiAssetHostModels<MarkovPack> host;
    host.models = {bs};
    host.scenario_indices = {0U};
    const std::vector<Rough::Parameters> rough_host{rough[0], rough[1]};
    const std::vector<std::string> internal_order{
        "asset_1.vol", "asset_2.vol", "asset_0.spot",
        "asset_1.spot", "asset_2.spot"
    };
    const std::vector<std::string> external_order{
        "asset_0.spot", "asset_1.spot", "asset_1.vol",
        "asset_2.spot", "asset_2.vol"
    };
    const correlation::MatrixRow internal_matrix{
        "000001", internal_order, lower
    };
    const std::vector<correlation::MatrixRow> matrices{
        correlation::reorder_factors(internal_matrix, external_order)
    };
    const auto volatility_first = offline::pricing::execute_mixed_rough_prices<
        MarkovPack, Rough, Product, 2U
    >(
        host, rough_host, {0U, 1U}, 1U, {product}, matrices,
        external_order, internal_order, {row}, paths, paths,
        1.0f / 252.0f, 1U, 32U, seed
    );
    const auto chunked = offline::pricing::execute_mixed_rough_prices<
        MarkovPack, Rough, Product, 2U
    >(
        host, rough_host, {0U, 1U}, 1U, {product}, matrices,
        external_order, internal_order, {row}, paths, 128U,
        1.0f / 252.0f, 1U, 32U, seed
    );
    double reference_mean = 0.0;
    for (float value : reference) reference_mean += value;
    reference_mean /= paths;
    require(std::abs(volatility_first.prices[0] - reference_mean) < 2.0e-6,
        "two rough host price disagrees with reference mean");
    require(std::abs(chunked.prices[0] - reference_mean) < 2.0e-6,
        "two rough chunked host price disagrees with reference mean");
    require(std::abs(volatility_first.standard_errors[0]
        - chunked.standard_errors[0]) < 2.0e-6,
        "two rough host standard errors disagree");
}
