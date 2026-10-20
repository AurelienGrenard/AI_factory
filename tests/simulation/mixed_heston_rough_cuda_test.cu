// Heston QE-M and causal rough Heston share four correlated Brownian factors.
#include "common/check_cuda.cuh"
#include "common/monte_carlo/multi_asset_rough_kernel.cuh"
#include "correlation/dataset.hpp"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/pricing/mixed_rough_price_generation.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace causal = ai_factory::workbench::volterra::causal_fft;
using Heston = model::equity::heston::DynamicsPolicy;
using Rough = model::equity::rough_heston::CausalFftPathPolicy;
using Product = product::WorstOfCallPathPolicy;
using MarkovPack = simulation::MultiAssetPack<
    simulation::ModelGroup<Heston, 1U>>;
constexpr unsigned paths = 512U;
constexpr unsigned steps = 96U;
constexpr std::uint64_t seed = 4839U;

__global__ void separate_reference(
    const Heston::Parameters* heston_parameters,
    const Rough::Parameters* rough_parameters,
    const float* lower,
    const float* volatility,
    float* payoffs
) {
    const auto path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    const auto key = philox::make_key(seed);
    const auto heston = Heston::prepare_dynamics(
        heston_parameters[0], 1.0f / 252.0f
    );
    auto state = Heston::initial_state(heston);
    for (unsigned step = 0U; step < steps; ++step) {
        simulation::BrownianNormals<4U, 1U> independent;
        independent.generate(key, path, step, 0U);
        simulation::CorrelatedBrownianStep<4U, 1U> brownian;
        brownian.calculate(independent, lower, 0U);
        philox::DomainUniformSequence auxiliary(
            key, path, (2U << 24U) | step
        );
        Heston::one_step_transition(heston,
            {brownian.get(1U, 0U), brownian.get(2U, 0U)},
            auxiliary, state);
    }
    const auto rough = Rough::prepare_model(
        rough_parameters[0], 1.0f / 252.0f
    );
    float log_spot = rough.initial_log_spot;
    for (unsigned step = 0U; step < steps; ++step) {
        simulation::BrownianNormals<4U, 1U> independent;
        independent.generate(key, path, step, 0U);
        simulation::CorrelatedBrownianStep<4U, 1U> brownian;
        brownian.calculate(independent, lower, 0U);
        const float variance = volatility[
            causal::history_index(path, step, steps)
        ];
        Rough::advance_spot_cell(rough, variance,
            brownian.get(3U, 0U), log_spot);
    }
    const float minimum = fminf(
        Heston::spot(state) / heston_parameters[0].spot,
        expf(log_spot - rough.initial_log_spot)
    );
    payoffs[path] = fmaxf(minimum - 0.9f, 0.0f);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    const Heston::Parameters heston{
        1.0f, 0.0f, 0.0f, 0.04f, 2.0f, 0.04f, 0.3f, -0.5f
    };
    const Rough::Parameters rough{
        1.0f, 0.0f, 0.0f, 0.05f, 2.0f, 0.1f, 0.3f, 0.12f, -0.4f
    };
    const Product::ProductParameters product{0.9f, steps};
    const monte_carlo::MultiAssetPriceRow row{0U, 0U, 0U, 0U, 1.0f};
    const std::vector<double> matrix{
        1.0, 0.1, 0.05, -0.4,
        0.1, 1.0, -0.5, 0.1,
        0.05, -0.5, 1.0, 0.0,
        -0.4, 0.1, 0.0, 1.0
    };
    const auto lower = correlation::packed_cholesky(matrix, 4U);
    const std::uint32_t index = 0U;
    offline::cuda::DeviceBuffer<Heston::Parameters> device_heston(1U);
    offline::cuda::DeviceBuffer<Rough::Parameters> device_rough(1U);
    offline::cuda::DeviceBuffer<std::uint32_t> device_heston_index(1U);
    offline::cuda::DeviceBuffer<std::uint32_t> device_rough_index(1U);
    offline::cuda::DeviceBuffer<Product::ProductParameters> device_product(1U);
    offline::cuda::DeviceBuffer<monte_carlo::MultiAssetPriceRow> device_row(1U);
    offline::cuda::DeviceBuffer<float> device_lower(lower.size());
    offline::cuda::DeviceBuffer<float> volatility(paths * steps);
    offline::cuda::DeviceBuffer<float> joint_payoffs(paths);
    offline::cuda::DeviceBuffer<float> reference_payoffs(paths);
    offline::cuda::DeviceBuffer<volterra::PartialMoments> moments(paths / 128U);
    const auto plan = causal::plan_volatility_workspace<Rough>(steps, paths);
    offline::cuda::DeviceBuffer<std::uint8_t> workspace(plan.bytes);
    device_heston.copy_from(&heston);
    device_rough.copy_from(&rough);
    device_heston_index.copy_from(&index);
    device_rough_index.copy_from(&index);
    device_product.copy_from(&product);
    device_row.copy_from(&row);
    device_lower.copy_from(lower.data());
    causal::launch_volatility_cuda<Rough, 4U, 0U, 3U>(
        device_rough.data(), device_rough_index.data(),
        device_row.data(), device_lower.data(), 0U, 0U,
        paths, 1.0f / 252.0f, steps, 32U, seed,
        workspace.data(), workspace.size(), volatility.data()
    );
    monte_carlo::MixedAssetDeviceInputs<MarkovPack, Rough, Product>
        inputs{{{device_heston.data(), device_heston_index.data()}, {}},
            device_rough.data(), device_rough_index.data(),
            device_product.data(), device_row.data(), device_lower.data()};
    monte_carlo::launch_joint_rough_cuda<MarkovPack, Rough, Product, true>(
        inputs, 0U, 0U, paths, steps, 1U, 1.0f / 252.0f, seed,
        volatility.data(), moments.data(), joint_payoffs.data()
    );
    separate_reference<<<paths / 128U, 128U>>>(
        device_heston.data(), device_rough.data(), device_lower.data(),
        volatility.data(), reference_payoffs.data()
    );
    check_cuda(cudaGetLastError(), "Heston rough reference launch");
    check_cuda(cudaDeviceSynchronize(), "Heston rough synchronize");
    std::vector<float> joint(paths), reference(paths);
    joint_payoffs.copy_to(joint.data());
    reference_payoffs.copy_to(reference.data());
    for (std::size_t path = 0U; path < paths; ++path) {
        require(std::isfinite(joint[path]), "Heston rough payoff not finite");
        require(std::abs(joint[path] - reference[path]) < 2.0e-6f,
            "joint QE Heston differs from separate marginal");
    }
    offline::pricing::MultiAssetHostModels<MarkovPack> host;
    host.models = {heston};
    host.scenario_indices = {0U};
    const std::vector<std::string> internal{
        "asset_1.vol", "asset_0.spot", "asset_0.vol", "asset_1.spot"
    };
    const std::vector<std::string> external{
        "asset_0.spot", "asset_0.vol", "asset_1.spot", "asset_1.vol"
    };
    const correlation::MatrixRow internal_matrix{
        "000001", internal, lower
    };
    const std::vector<correlation::MatrixRow> correlations{
        correlation::reorder_factors(internal_matrix, external)
    };
    const auto run = offline::pricing::execute_mixed_rough_prices<
        MarkovPack, Rough, Product
    >(
        host, {rough}, {0U}, 1U, {product}, correlations,
        external, internal, {row}, paths, 128U,
        1.0f / 252.0f, 1U, 32U, seed
    );
    double reference_mean = 0.0;
    for (float payoff : reference) reference_mean += payoff;
    reference_mean /= paths;
    require(std::abs(run.prices[0] - reference_mean) < 2.0e-6,
        "Heston rough host price differs from reference mean");
}
