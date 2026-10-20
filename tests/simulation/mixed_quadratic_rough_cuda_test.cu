// Joint BS/rough-Heston payoff versus separately advanced marginals.
#include "common/check_cuda.cuh"
#include "common/monte_carlo/multi_asset_rough_kernel.cuh"
#include "common/volterra/causal_fft_volatility.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_dynamics.cuh"
#include "product/worst_of_call/pricing_policy.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace causal = ai_factory::workbench::volterra::causal_fft;
using BS = model::equity::black_scholes::DynamicsPolicy;
using Rough = model::equity::quadratic_rough_heston::CausalFftPathPolicy;
using Product = product::WorstOfCallPathPolicy;
using MarkovPack = simulation::MultiAssetPack<simulation::ModelGroup<BS, 1U>>;
constexpr unsigned steps = 128U;
constexpr unsigned paths = 256U;
constexpr std::uint64_t seed = 8121009ULL;

__global__ void separate_marginals_reference(
    const BS::Parameters* bs_models,
    const Rough::Parameters* rough_models,
    const float* lower,
    const float* volatility_path,
    float* payoffs
) {
    const auto path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    const auto key = philox::make_key(seed);
    const auto bs = BS::prepare_dynamics(bs_models[0], 1.0f / 252.0f);
    auto bs_state = BS::initial_state(bs);
    for (unsigned step = 0U; step < steps; ++step) {
        simulation::BrownianNormals<2U, 1U> independent;
        independent.generate(key, path, step, 0U);
        simulation::CorrelatedBrownianStep<2U, 1U> correlated;
        correlated.calculate(independent, lower, 0U);
        philox::DomainUniformSequence auxiliary(
            key, path, (2U << 24U) | step
        );
        BS::one_step_transition(
            bs, BS::BrownianStep{correlated.get(1U, 0U)},
            auxiliary, bs_state
        );
    }
    const auto rough = Rough::prepare_model(
        rough_models[0], 1.0f / 252.0f
    );
    float rough_log_spot = rough.initial_log_spot;
    for (unsigned step = 0U; step < steps; ++step) {
        simulation::BrownianNormals<2U, 1U> independent;
        independent.generate(key, path, step, 0U);
        simulation::CorrelatedBrownianStep<2U, 1U> correlated;
        correlated.calculate(independent, lower, 0U);
        Rough::advance_spot_cell(
            rough, volatility_path[causal::history_index(path, step, steps)],
            correlated.get(0U, 0U), rough_log_spot
        );
    }
    const float minimum = fminf(
        BS::spot(bs_state) / bs_models[0].spot,
        expf(rough_log_spot - rough.initial_log_spot)
    );
    payoffs[path] = fmaxf(minimum - 0.9f, 0.0f);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    const BS::Parameters bs{1.0f, 0.0f, 0.0f, 0.2f};
    const Rough::Parameters rough{
        1.0f, 0.0f, 0.0f, 0.05f, 0.7f, 0.1f,
        0.02f, 1.4f, 0.3f, 0.12f
    };
    const Product::ProductParameters product{0.9f, steps};
    const monte_carlo::MultiAssetPriceRow row{0U, 0U, 0U, 0U, 1.0f};
    // The rough feedback and rough spot share factor zero.
    const float lower[3]{1.0f, 0.2f, 0.9797959f};
    const std::uint32_t index = 0U;
    BS::Parameters* device_bs = nullptr;
    Rough::Parameters* device_rough = nullptr;
    std::uint32_t *device_bs_index = nullptr, *device_rough_index = nullptr;
    Product::ProductParameters* device_product = nullptr;
    monte_carlo::MultiAssetPriceRow* device_row = nullptr;
    float *device_lower = nullptr, *device_volatility = nullptr;
    float *device_joint = nullptr, *device_separate = nullptr;
    float *device_price = nullptr, *device_error = nullptr;
    volterra::PartialMoments* device_moments = nullptr;
    void* workspace = nullptr;
    const auto plan = causal::plan_volatility_workspace<Rough>(steps, paths);
    check_cuda(cudaMalloc(&device_bs, sizeof(bs)), "mixed BS allocation");
    check_cuda(cudaMalloc(&device_rough, sizeof(rough)), "mixed rough allocation");
    check_cuda(cudaMalloc(&device_bs_index, sizeof(index)), "mixed BS index allocation");
    check_cuda(cudaMalloc(&device_rough_index, sizeof(index)), "mixed rough index allocation");
    check_cuda(cudaMalloc(&device_product, sizeof(product)), "mixed product allocation");
    check_cuda(cudaMalloc(&device_row, sizeof(row)), "mixed row allocation");
    check_cuda(cudaMalloc(&device_lower, sizeof(lower)), "mixed factor allocation");
    check_cuda(cudaMalloc(&device_volatility, paths * steps * sizeof(float)),
        "mixed variance allocation");
    check_cuda(cudaMalloc(&device_joint, paths * sizeof(float)),
        "mixed joint payoff allocation");
    check_cuda(cudaMalloc(&device_separate, paths * sizeof(float)),
        "mixed separate payoff allocation");
    check_cuda(cudaMalloc(&device_moments, 2U * sizeof(volterra::PartialMoments)),
        "mixed moments allocation");
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "mixed price allocation");
    check_cuda(cudaMalloc(&device_error, sizeof(float)), "mixed error allocation");
    check_cuda(cudaMalloc(&workspace, plan.bytes), "mixed workspace allocation");
    check_cuda(cudaMemcpy(device_bs, &bs, sizeof(bs), cudaMemcpyHostToDevice),
        "mixed BS upload");
    check_cuda(cudaMemcpy(device_rough, &rough, sizeof(rough), cudaMemcpyHostToDevice),
        "mixed rough upload");
    check_cuda(cudaMemcpy(device_bs_index, &index, sizeof(index), cudaMemcpyHostToDevice),
        "mixed BS index upload");
    check_cuda(cudaMemcpy(device_rough_index, &index, sizeof(index), cudaMemcpyHostToDevice),
        "mixed rough index upload");
    check_cuda(cudaMemcpy(device_product, &product, sizeof(product), cudaMemcpyHostToDevice),
        "mixed product upload");
    check_cuda(cudaMemcpy(device_row, &row, sizeof(row), cudaMemcpyHostToDevice),
        "mixed row upload");
    check_cuda(cudaMemcpy(device_lower, lower, sizeof(lower), cudaMemcpyHostToDevice),
        "mixed factor upload");

    causal::launch_volatility_cuda<Rough, 2U, 0U, 3U>(
        device_rough, device_rough_index, device_row, device_lower,
        0U, 0U, paths, 1.0f / 252.0f, steps, 32U, seed,
        workspace, plan.bytes, device_volatility
    );
    monte_carlo::MixedAssetDeviceInputs<MarkovPack, Rough, Product> inputs{
        {{device_bs, device_bs_index}, {}},
        device_rough, device_rough_index, device_product,
        device_row, device_lower
    };
    monte_carlo::launch_joint_rough_cuda<
        MarkovPack, Rough, Product, true
    >(
        inputs, 0U, 0U, paths, steps, 1U, 1.0f / 252.0f,
        seed, device_volatility, device_moments, device_joint
    );
    separate_marginals_reference<<<2U, 128U>>>(
        device_bs, device_rough, device_lower,
        device_volatility, device_separate
    );
    check_cuda(cudaGetLastError(), "separate marginals launch");
    volterra::finalize_price_kernel<<<1U, 128U,
        2U * (128U / 32U) * sizeof(double)>>>(
            device_moments, 2U, paths, 0U,
            device_price, device_error
        );
    check_cuda(cudaGetLastError(), "mixed price finalize");
    check_cuda(cudaDeviceSynchronize(), "mixed test synchronize");
    std::vector<float> joint(paths), separate(paths);
    check_cuda(cudaMemcpy(joint.data(), device_joint, paths * sizeof(float),
        cudaMemcpyDeviceToHost), "joint payoff download");
    check_cuda(cudaMemcpy(separate.data(), device_separate, paths * sizeof(float),
        cudaMemcpyDeviceToHost), "separate payoff download");
    float max_error = 0.0f;
    for (std::size_t i = 0U; i < paths; ++i) {
        require(std::isfinite(joint[i]), "nonfinite mixed payoff");
        max_error = std::max(max_error, std::abs(joint[i] - separate[i]));
    }
    require(max_error < 2.0e-6f,
        "joint paths disagree with separately simulated marginals");
    float price = 0.0f;
    check_cuda(cudaMemcpy(&price, device_price, sizeof(price),
        cudaMemcpyDeviceToHost), "mixed price download");
    double mean = 0.0;
    for (float payoff : joint) mean += payoff;
    mean /= paths;
    require(std::abs(static_cast<double>(price) - mean) < 2.0e-6,
        "mixed price disagrees with pathwise mean");
    cudaFree(device_bs);
    cudaFree(device_rough);
    cudaFree(device_bs_index);
    cudaFree(device_rough_index);
    cudaFree(device_product);
    cudaFree(device_row);
    cudaFree(device_lower);
    cudaFree(device_volatility);
    cudaFree(device_joint);
    cudaFree(device_separate);
    cudaFree(device_moments);
    cudaFree(device_price);
    cudaFree(device_error);
    cudaFree(workspace);
}
