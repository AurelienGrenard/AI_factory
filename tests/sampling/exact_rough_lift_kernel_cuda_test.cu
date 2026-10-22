// The sample kernels fit each sampled H on the device, using the CPU L2 rule.
#include "cuda/check_cuda.cuh"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/rough_heston/sample_device_preparation.cuh"
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/quadratic_rough_heston/sample_device_preparation.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
namespace rh = ai_factory::workbench::model::equity::rough_heston;
namespace qrh = ai_factory::workbench::model::equity::quadratic_rough_heston;
using ai_factory::workbench::check_cuda;
constexpr float kDt = 1.0f / 504.0f;
constexpr float kHorizon = 2.0f;

void require_close(float actual, float reference) {
    if (!std::isfinite(actual) || !std::isfinite(reference)
        || std::abs(actual - reference)
            > 2.0e-6f + 5.0e-5f * std::abs(reference)) {
        throw std::runtime_error("Device fitted lift differs from CPU L2 fit.");
    }
}

template<typename Parameters, typename Prepared, typename Preparer>
__global__ void prepare_on_device(
    const Parameters* models, Prepared* outputs, std::size_t count
) {
    const std::size_t index = threadIdx.x;
    if (index < count) {
        outputs[index] = Preparer::prepare(models[index], {kHorizon, kDt});
    }
}

template<typename Parameters, typename Prepared, typename Preparer>
std::vector<Prepared> device_prepare(const std::vector<Parameters>& models) {
    Parameters* device_models = nullptr;
    Prepared* device_outputs = nullptr;
    check_cuda(cudaMalloc(&device_models, models.size() * sizeof(Parameters)),
               "allocate exact lift parameters");
    check_cuda(cudaMalloc(&device_outputs, models.size() * sizeof(Prepared)),
               "allocate exact lift outputs");
    check_cuda(cudaMemcpy(device_models, models.data(),
                          models.size() * sizeof(Parameters),
                          cudaMemcpyHostToDevice), "copy exact lift parameters");
    prepare_on_device<Parameters, Prepared, Preparer><<<1, 32>>>(
        device_models, device_outputs, models.size()
    );
    check_cuda(cudaDeviceSynchronize(), "prepare exact lift on device");
    std::vector<Prepared> result(models.size());
    check_cuda(cudaMemcpy(result.data(), device_outputs,
                          result.size() * sizeof(Prepared),
                          cudaMemcpyDeviceToHost), "copy exact lift outputs");
    check_cuda(cudaFree(device_outputs), "free exact lift outputs");
    check_cuda(cudaFree(device_models), "free exact lift parameters");
    return result;
}

template<typename Prepared>
void compare_kernels(const Prepared& actual, const Prepared& expected) {
    for (std::size_t factor = 0; factor < 7U; ++factor) {
        require_close(actual.kernel.nodes[factor], expected.kernel.nodes[factor]);
        require_close(actual.kernel.weights[factor], expected.kernel.weights[factor]);
    }
}

void test_rough_heston() {
    std::vector<rh::ModelParameters> models;
    for (const float hurst : {0.03f, 0.118f, 0.25f}) {
        models.push_back({1.0f, 0.0f, 0.0f, 0.04f, 1.5f, 0.06f,
                          0.30f, hurst, -0.70f});
    }
    const auto actual = device_prepare<rh::ModelParameters,
        rh::PreparedDynamics<7U>, rh::SampleDevicePreparer<7U>>(models);
    for (std::size_t index = 0; index < models.size(); ++index) {
        const auto expected = rh::prepare_dynamics<7U>(
            models[index], kHorizon, kDt
        );
        compare_kernels(actual[index], expected);
        for (std::size_t factor = 0; factor < 7U; ++factor) {
            require_close(actual[index].initial_factors[factor],
                          expected.initial_factors[factor]);
            require_close(actual[index].ode_half_shift[factor],
                          expected.ode_half_shift[factor]);
        }
        for (std::size_t entry = 0; entry < 49U; ++entry) {
            require_close(actual[index].ode_half_step[entry],
                          expected.ode_half_step[entry]);
        }
    }
}

void test_quadratic_rough_heston() {
    std::vector<qrh::ModelParameters> models;
    for (const float hurst : {0.05f, 0.125f, 0.20f}) {
        models.push_back({1.0f, 0.0f, 0.0f, 0.03f, 0.25f, 0.10f,
                          0.005f, 1.0f, 0.50f, hurst});
    }
    const auto actual = device_prepare<qrh::ModelParameters,
        qrh::PreparedDynamics<7U>, qrh::SampleDevicePreparer<7U>>(models);
    for (std::size_t index = 0; index < models.size(); ++index) {
        const auto expected = qrh::prepare_dynamics<7U>(
            models[index], kHorizon, kDt
        );
        compare_kernels(actual[index], expected);
        for (std::size_t factor = 0; factor < 7U; ++factor) {
            require_close(actual[index].factor_decay[factor],
                          expected.factor_decay[factor]);
            require_close(actual[index].factor_drift_integral[factor],
                          expected.factor_drift_integral[factor]);
        }
        require_close(actual[index].feedback_cell_loading,
                      expected.feedback_cell_loading);
    }
}
}  // namespace

int main() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || count == 0) return 77;
    check_cuda(status, "detect exact lift GPU");
    test_rough_heston();
    test_quadratic_rough_heston();
}
