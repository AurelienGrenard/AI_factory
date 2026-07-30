// Validate Nelson-Siegel and exact Hull-White building blocks on CUDA.
#include "common/check_cuda.cuh"

// Include the implementation exactly as future product kernels will.
#include "model/hull_white/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr std::size_t kOutputCount = 16U;

// Evaluate curve limits, one exact transition, and one model zero-coupon.
__global__ void hull_white_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    const ai_factory::workbench::curve::NelsonSiegelParameters initial_curve = {
        0.03f,
        -0.01f,
        0.02f,
        2.0f,
    };
    const ai_factory::workbench::hull_white::HullWhiteOneFactorParameters
        model = {0.15f, 0.01f};
    constexpr float dt = 1.0f / 12.0f;
    constexpr float maturity = 5.0f;

    using namespace ai_factory::workbench;
    hull_white::HullWhiteState state =
        hull_white::initial_hull_white_state();
    const hull_white::HullWhiteStepParameters step =
        hull_white::prepare_hull_white_step(model, dt);

    outputs[0] = curve::nelson_siegel_zero_rate(initial_curve, 0.0f);
    outputs[1] = initial_curve.beta0 + initial_curve.beta1;
    outputs[2] = hull_white::hull_white_zero_coupon_bond(
        model, initial_curve, state, 0.0f, maturity
    );
    outputs[3] =
        curve::nelson_siegel_discount_factor(initial_curve, maturity);
    outputs[4] = step.decay;
    outputs[5] = step.factor_standard_deviation;

    hull_white::one_step_hull_white_transition(
        step, 0.0f, 0.0f, state
    );
    outputs[6] = state.factor;
    outputs[7] = hull_white::hull_white_log_discount(
        model, initial_curve, state, dt
    );
    outputs[8] = hull_white::hull_white_zero_coupon_bond(
        model, initial_curve, state, dt, maturity
    );
    outputs[9] = hull_white::hull_white_short_rate(
        model, initial_curve, state, dt
    );

    const philox::PhiloxKey key = philox::make_key(900000001ULL);
    const hull_white::HullWhiteState terminal =
        hull_white::simulate_terminal_hull_white_state(
            step, key, 0U, 12U
        );
    outputs[10] = terminal.factor;
    outputs[11] = terminal.integrated_factor;

    float observed_factors[2] = {};
    float observed_integrated_factors[2] = {};
    const hull_white::HullWhiteState grid_terminal =
        hull_white::simulate_factor_discount_on_regular_grid(
            step,
            step,
            key,
            0U,
            1U,
            1U,
            3U,
            1U,
            observed_factors,
            observed_integrated_factors
        );
    outputs[12] = observed_factors[0];
    outputs[13] = observed_integrated_factors[0];
    outputs[14] = grid_terminal.factor;
    outputs[15] = grid_terminal.integrated_factor;
}

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

// Confirm that the reusable fixed-income CUDA primitives are coherent.
int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "Hull-White test cudaGetDeviceCount");

    float* device_outputs = nullptr;
    check_cuda(
        cudaMalloc(&device_outputs, kOutputCount * sizeof(float)),
        "Hull-White test cudaMalloc"
    );

    float outputs[kOutputCount] = {};
    try {
        hull_white_test_kernel<<<1U, 1U>>>(device_outputs);
        check_cuda(cudaGetLastError(), "Hull-White test kernel launch");
        check_cuda(
            cudaMemcpy(
                outputs,
                device_outputs,
                kOutputCount * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Hull-White test cudaMemcpy"
        );
        check_cuda(cudaFree(device_outputs), "Hull-White test cudaFree");
        device_outputs = nullptr;
    } catch (...) {
        if (device_outputs != nullptr) cudaFree(device_outputs);
        throw;
    }

    require(
        std::fabs(outputs[0] - outputs[1]) < 1.0e-6f,
        "Nelson-Siegel zero-maturity limit is incorrect"
    );
    require(
        std::fabs(outputs[2] - outputs[3]) < 2.0e-6f,
        "Hull-White time-zero bond does not reproduce the initial curve"
    );
    require(
        outputs[4] > 0.0f && outputs[4] < 1.0f && outputs[5] > 0.0f,
        "Hull-White exact transition coefficients are invalid"
    );
    require(
        outputs[6] == 0.0f && outputs[7] < 0.0f,
        "Hull-White zero-normal transition is invalid"
    );
    require(
        std::isfinite(outputs[8]) && outputs[8] > 0.0f
            && std::isfinite(outputs[9]),
        "Hull-White post-transition values are invalid"
    );
    for (std::size_t index = 10U; index < kOutputCount; ++index) {
        require(
            std::isfinite(outputs[index]),
            "Hull-White path simulation returned a non-finite state"
        );
    }
}
