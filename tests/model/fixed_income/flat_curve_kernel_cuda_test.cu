// Validate the flat term structure and its exact Nelson-Siegel embedding on CUDA.
#include "common/check_cuda.cuh"
#include "curve/flat/term_structure_impl.cuh"
#include "curve/nelson_siegel/term_structure_impl.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

namespace flat = ai_factory::workbench::curve::flat;
namespace ns = ai_factory::workbench::curve::nelson_siegel;

constexpr std::array<float, 3U> kRates = {0.03f, 0.0f, -0.01f};
constexpr std::array<float, 4U> kMaturities = {0.0f, 0.25f, 1.0f, 10.0f};
constexpr std::size_t kQuantityCount = 7U;
constexpr std::size_t kOutputCount =
    kRates.size() * kMaturities.size() * kQuantityCount;

__global__ void flat_curve_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    const float rates[] = {0.03f, 0.0f, -0.01f};
    const float maturities[] = {0.0f, 0.25f, 1.0f, 10.0f};
    std::size_t output = 0U;
    for (const float rate : rates) {
        const flat::FlatCurveParameters curve{rate};
        const ns::NelsonSiegelParameters embedded_curve{
            rate, 0.0f, 0.0f, 1.0f
        };
        for (const float maturity : maturities) {
            outputs[output++] = flat::zero_rate(curve, maturity);
            outputs[output++] = flat::log_discount_factor(curve, maturity);
            outputs[output++] = flat::discount_factor(curve, maturity);
            outputs[output++] = flat::instantaneous_forward(curve, maturity);
            outputs[output++] = flat::forward_derivative(curve, maturity);
            outputs[output++] = flat::forward_rate(
                curve, maturity, maturity + 0.5f
            );
            outputs[output++] = ns::discount_factor(
                embedded_curve, maturity
            );
        }
    }
}

void require_close(float actual, float expected, const char* message) {
    if (!std::isfinite(actual)
        || std::fabs(actual - expected) > 2.0e-6f) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "Flat-curve test cudaGetDeviceCount");

    float* device_outputs = nullptr;
    check_cuda(
        cudaMalloc(&device_outputs, kOutputCount * sizeof(float)),
        "Flat-curve test cudaMalloc"
    );
    std::array<float, kOutputCount> outputs{};
    flat_curve_test_kernel<<<1U, 1U>>>(device_outputs);
    check_cuda(cudaGetLastError(), "Flat-curve test kernel");
    check_cuda(
        cudaMemcpy(
            outputs.data(),
            device_outputs,
            kOutputCount * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "Flat-curve test cudaMemcpy"
    );
    check_cuda(cudaFree(device_outputs), "Flat-curve test cudaFree");

    std::size_t output = 0U;
    for (const float rate : kRates) {
        for (const float maturity : kMaturities) {
            const float expected_log_discount = -rate * maturity;
            const float expected_discount = std::exp(expected_log_discount);
            require_close(outputs[output++], rate, "Flat zero rate mismatch");
            require_close(
                outputs[output++],
                expected_log_discount,
                "Flat log-discount mismatch"
            );
            require_close(
                outputs[output++], expected_discount, "Flat discount mismatch"
            );
            require_close(
                outputs[output++], rate, "Flat instantaneous forward mismatch"
            );
            require_close(
                outputs[output++], 0.0f, "Flat forward derivative mismatch"
            );
            require_close(
                outputs[output++], rate, "Flat finite-period forward mismatch"
            );
            require_close(
                outputs[output++],
                expected_discount,
                "Flat curve does not match its Nelson-Siegel embedding"
            );
        }
    }
}
