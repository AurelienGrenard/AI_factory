// Causal volatility-only FFT versus direct convolution with identical factors.
#include "common/check_cuda.cuh"
#include "common/volterra/causal_fft_volatility_kernel.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_dynamics.cuh"

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
constexpr unsigned steps = 128U;
constexpr unsigned paths = 256U;
constexpr std::uint64_t seed = 923184ULL;

template<typename Policy>
__global__ void direct_reference(
    const typename Policy::Parameters* parameters,
    const float* packed_lower,
    float* volatility
) {
    const auto path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    const auto model = Policy::prepare_model(parameters[0], 1.0f / 252.0f);
    const auto kernel = Policy::KernelPolicy::prepare(
        Policy::kernel_parameters(parameters[0]), 1.0f / 252.0f
    );
    auto state = Policy::initial_volatility_state(model);
    const auto key = philox::make_key(seed);
    float sources[steps];
    for (unsigned step = 0U; step < steps; ++step) {
        float far = 0.0f;
        for (unsigned prior = 0U; prior < step; ++prior) {
            const float weight = Policy::KernelPolicy::cell_average_weight(
                kernel, step - prior
            );
            far = fmaf(weight, sources[prior], far);
        }
        const float normal = simulation::correlated_brownian_factor(
            key, path, step, 0U, packed_lower
        );
        float singular = 0.0f;
        if constexpr (Policy::kBrownianDimension == 2U) {
            const auto uniforms = philox::domain_uniform_quad(
                key, path, 0U, (3U << 24U) | step
            );
            singular = philox::box_muller(
                uniforms.first, uniforms.second
            ).first;
        }
        float variance = 0.0f;
        sources[step] = Policy::advance_volatility_cell(
            model, kernel, far, normal, singular, step, state, variance
        );
        volatility[causal::history_index(path, step, steps)] = variance;
    }
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<typename Policy>
void exercise(const typename Policy::Parameters& parameters) {
    using Model = typename Policy::Parameters;
    Model* device_model = nullptr;
    RowIndexSelector* device_selectors = nullptr;
    float *device_lower = nullptr, *device_output = nullptr;
    float* device_reference = nullptr;
    void* workspace = nullptr;
    const auto plan = causal::plan_volatility_workspace<Policy>(steps, paths);
    check_cuda(cudaMalloc(&device_model, sizeof(Model)), "vol model allocation");
    check_cuda(cudaMalloc(&device_selectors, 2U * sizeof(RowIndexSelector)),
               "vol selector allocation");
    check_cuda(cudaMalloc(&device_lower, 6U * sizeof(float)), "vol lower allocation");
    check_cuda(cudaMalloc(&device_output, paths * steps * sizeof(float)),
               "vol output allocation");
    check_cuda(cudaMalloc(&device_reference, paths * steps * sizeof(float)),
               "vol reference allocation");
    check_cuda(cudaMalloc(&workspace, plan.bytes), "vol workspace allocation");
    const RowIndexSelector selectors[2]{{1U,1U,RowIndexMode::Strided},
                                        {1U,1U,RowIndexMode::Strided}};
    const float lower[6]{1.0f, 0.2f, 0.9797959f,
                         -0.4f, 0.1f, 0.9110433f};
    check_cuda(cudaMemcpy(device_model, &parameters, sizeof(Model),
                          cudaMemcpyHostToDevice), "vol model upload");
    check_cuda(cudaMemcpy(device_selectors, selectors, sizeof(selectors),
                          cudaMemcpyHostToDevice), "vol selector upload");
    check_cuda(cudaMemcpy(device_lower, lower, sizeof(lower),
                          cudaMemcpyHostToDevice), "vol factor upload");
    causal::launch_volatility_factor_cuda<Policy, 3U>(
        device_model, device_lower,0U,0U,paths,1.0f/252.0f,steps,32U,
        seed,workspace,plan.bytes,device_output,0U,3U,0,
        device_selectors,0U,1U
    );
    direct_reference<Policy><<<2U, 128U>>>(
        device_model, device_lower, device_reference
    );
    check_cuda(cudaGetLastError(), "vol direct reference launch");
    check_cuda(cudaDeviceSynchronize(), "vol test synchronize");
    std::vector<float> actual(paths * steps), expected(paths * steps);
    check_cuda(cudaMemcpy(actual.data(), device_output,
        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "vol output download");
    check_cuda(cudaMemcpy(expected.data(), device_reference,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "vol reference download");
    float maximum = 0.0f;
    for (std::size_t i = 0U; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "nonfinite causal variance");
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    require(maximum < 2.0e-4f, "causal volatility differs from direct reference");
    cudaFree(device_model);
    cudaFree(device_selectors);
    cudaFree(device_lower);
    cudaFree(device_output);
    cudaFree(device_reference);
    cudaFree(workspace);
}
}

int main() {
    exercise<model::equity::rough_heston::CausalFftPathPolicy>({
        1.0f, 0.0f, 0.0f, 0.04f, 2.0f, 0.08f, 0.4f, 0.12f, -0.4f
    });
    exercise<model::equity::quadratic_rough_heston::CausalFftPathPolicy>({
        1.0f, 0.0f, 0.0f, 0.05f, 0.2f, 0.0f, 0.04f, 1.0f, 0.3f, 0.12f
    });
}
