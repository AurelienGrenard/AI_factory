// Large factor-count mixed path: one rough Heston and 30 BS positions.
#include "common/check_cuda.cuh"
#include "common/monte_carlo/multi_asset_rough_kernel.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/cuda/pricing_runner.cuh"

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
using Product = product::WorstOfCallPathPolicy;
using MarkovPack = simulation::MultiAssetPack<simulation::ModelGroup<BS, 30U>>;
constexpr unsigned paths = 1024U;
constexpr unsigned steps = 64U;
constexpr std::uint64_t seed = 7331U;

__global__ void separate_reference(
    const BS::Parameters* bs_parameters,
    const Rough::Parameters* rough_parameters,
    const float* lower,
    const float* variance,
    float* payoffs
) {
    const auto path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    const auto key = philox::make_key(seed);
    const auto bs = BS::prepare_dynamics(bs_parameters[0], 1.0f / 252.0f);
    float minimum = __int_as_float(0x7f800000);
    for (unsigned asset = 0U; asset < 30U; ++asset) {
        auto state = BS::initial_state(bs);
        for (unsigned step = 0U; step < steps; ++step) {
            const float normal = simulation::correlated_brownian_factor(
                key, path, step, asset + 1U, lower
            );
            philox::DomainUniformSequence auxiliary(
                key, path, ((2U + asset) << 24U) | step
            );
            BS::one_step_transition(bs, {normal}, auxiliary, state);
        }
        minimum = fminf(minimum,
            BS::spot(state) / bs_parameters[0].spot);
    }
    const auto rough = Rough::prepare_model(
        rough_parameters[0], 1.0f / 252.0f
    );
    float log_spot = rough.initial_log_spot;
    for (unsigned step = 0U; step < steps; ++step) {
        const float normal = simulation::correlated_brownian_factor(
            key, path, step, 31U, lower
        );
        const float value = variance[causal::history_index(path, step, steps)];
        Rough::advance_spot_cell(rough, value, normal, log_spot);
    }
    minimum = fminf(minimum,
        expf(log_spot - rough.initial_log_spot));
    payoffs[path] = minimum;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    const BS::Parameters bs{1.0f, 0.0f, 0.0f, 0.2f};
    const Rough::Parameters rough{
        1.0f, 0.0f, 0.0f, 0.04f, 2.0f, 0.08f, 0.3f, 0.12f, -0.4f
    };
    const Product::ProductParameters product{0.0f, steps};
    const monte_carlo::MultiAssetPriceRow row{0U, 0U, 0U, 0U, 1.0f};
    std::vector<std::uint32_t> indices(30U, 0U);
    std::vector<float> lower(32U * 33U / 2U, 0.0f);
    for (std::size_t i = 0U; i < 32U; ++i)
        lower[i * (i + 1U) / 2U + i] = 1.0f;
    lower[31U * 32U / 2U] = -0.4f;
    lower[31U * 32U / 2U + 31U] = std::sqrt(0.84f);
    offline::cuda::DeviceBuffer<BS::Parameters> device_bs(1U);
    offline::cuda::DeviceBuffer<Rough::Parameters> device_rough(1U);
    offline::cuda::DeviceBuffer<std::uint32_t> device_indices(indices.size());
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
    const std::uint32_t index = 0U;
    device_bs.copy_from(&bs);
    device_rough.copy_from(&rough);
    device_indices.copy_from(indices.data());
    device_rough_index.copy_from(&index);
    device_product.copy_from(&product);
    device_row.copy_from(&row);
    device_lower.copy_from(lower.data());
    causal::launch_volatility_cuda<Rough, 32U, 0U, 32U>(
        device_rough.data(), device_rough_index.data(),
        device_row.data(), device_lower.data(), 0U, 0U,
        paths, 1.0f / 252.0f, steps, 32U, seed,
        workspace.data(), workspace.size(), volatility.data()
    );
    monte_carlo::MixedAssetDeviceInputs<MarkovPack, Rough, Product>
        inputs{{{device_bs.data(), device_indices.data()}, {}},
            device_rough.data(), device_rough_index.data(),
            device_product.data(), device_row.data(), device_lower.data()};
    const auto launch_joint = [&] {
        monte_carlo::launch_joint_rough_cuda<
            MarkovPack, Rough, Product, true
        >(
            inputs, 0U, 0U, paths, steps, 1U, 1.0f / 252.0f, seed,
            volatility.data(), moments.data(), joint_payoffs.data()
        );
    };
    const auto launch_separate = [&] {
        separate_reference<<<paths / 128U, 128U>>>(
            device_bs.data(), device_rough.data(), device_lower.data(),
            volatility.data(), reference_payoffs.data()
        );
        check_cuda(cudaGetLastError(), "large dimension reference launch");
    };
    launch_joint();
    launch_separate();
    check_cuda(cudaDeviceSynchronize(), "large dimension synchronize");
    std::vector<float> joint(paths), reference(paths);
    joint_payoffs.copy_to(joint.data());
    reference_payoffs.copy_to(reference.data());
    for (std::size_t path = 0U; path < paths; ++path) {
        require(std::isfinite(joint[path]), "large dimension payoff not finite");
        require(std::abs(joint[path] - reference[path]) < 2.0e-5f,
            "large dimension joint and separate paths disagree");
    }
    offline::cuda::Event start, stop;
    auto time_kernel = [&](auto launch) {
        launch();
        check_cuda(cudaDeviceSynchronize(), "large dimension warmup");
        check_cuda(cudaEventRecord(start.get()), "large dimension timer start");
        for (unsigned repeat = 0U; repeat < 10U; ++repeat) launch();
        check_cuda(cudaEventRecord(stop.get()), "large dimension timer stop");
        check_cuda(cudaEventSynchronize(stop.get()), "large dimension timer wait");
        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
            "large dimension timer elapsed");
        return milliseconds / 10.0f;
    };
    const auto launch_volatility_first = [&] {
        causal::launch_volatility_cuda<Rough, 32U, 0U, 32U>(
            device_rough.data(), device_rough_index.data(),
            device_row.data(), device_lower.data(), 0U, 0U,
            paths, 1.0f / 252.0f, steps, 32U, seed,
            workspace.data(), workspace.size(), volatility.data()
        );
        launch_joint();
    };
    std::cout << "large_joint_scan_ms=" << time_kernel(launch_joint)
              << " large_separate_scan_ms=" << time_kernel(launch_separate)
              << " large_volatility_first_full_ms="
              << time_kernel(launch_volatility_first) << '\n';
}
