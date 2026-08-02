// Validate Nelson-Siegel and exact Hull-White building blocks on CUDA.
#include "common/check_cuda.cuh"

// Include the implementation exactly as future product kernels will.
#include "model/hull_white/nelson_siegel/analytics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

namespace fitted =
    ai_factory::workbench::model::hull_white::nelson_siegel;
namespace ou =
    ai_factory::workbench::model::ornstein_uhlenbeck;

constexpr std::size_t kOutputCount = 30U;

// Evaluate curve limits, one exact transition, and one model zero-coupon.
__global__ void hull_white_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    const ai_factory::workbench::curve::nelson_siegel::NelsonSiegelParameters
        initial_curve = {
        0.03f,
        -0.01f,
        0.02f,
        2.0f,
    };
    const ai_factory::workbench::model::hull_white::HullWhiteModelParameters
        model = {0.15f, 0.01f};
    constexpr float dt = 1.0f / 12.0f;
    constexpr float maturity = 5.0f;

    using namespace ai_factory::workbench;
    const fitted::HullWhiteFittedParameters prepared =
        fitted::compose_model(model, initial_curve);
    const ou::OrnsteinUhlenbeckExactTransition step =
        ou::prepare_model(prepared.process, dt);
    const ou::joint::OrnsteinUhlenbeckJointExactTransition joint_step =
        ou::joint::prepare_model(prepared.process, dt);
    const ou::OrnsteinUhlenbeckExactTransition terminal_step =
        ou::prepare_model(prepared.process, 12.0f * dt);
    const ou::joint::OrnsteinUhlenbeckJointExactTransition
        joint_terminal_step =
            ou::joint::prepare_model(prepared.process, 12.0f * dt);
    float state = 0.0f;
    ou::joint::OrnsteinUhlenbeckJointState joint_state{0.0f, 0.0f};

    outputs[0] = curve::nelson_siegel::zero_rate(initial_curve, 0.0f);
    outputs[1] = initial_curve.beta0 + initial_curve.beta1;
    outputs[2] = fitted::zero_coupon_bond(
        prepared, state, 0.0f, maturity
    );
    outputs[3] =
        curve::nelson_siegel::discount_factor(initial_curve, maturity);
    outputs[4] = step.decay;
    outputs[5] = step.state_standard_deviation;

    ou::one_step_transition(step, 0.0f, state);
    ou::joint::one_step_transition(joint_step, 0.0f, 0.0f, joint_state);
    outputs[6] = state;
    outputs[7] = fitted::log_discount_factor(prepared, joint_state, dt);
    outputs[8] = fitted::zero_coupon_bond(
        prepared, state, dt, maturity
    );
    outputs[9] = fitted::short_rate(prepared, state, dt);

    const philox::PhiloxKey key = philox::make_key(900000001ULL);
    const ou::joint::OrnsteinUhlenbeckJointState terminal =
        ou::joint::simulate_terminal_state(
            joint_terminal_step, 0.0f, 0.25f, -0.75f
        );
    outputs[10] = terminal.state;
    outputs[11] = terminal.state_integral;

    float observed_states[2] = {};
    float observed_integrated_states[2] = {};
    const ou::joint::OrnsteinUhlenbeckJointState grid_terminal =
        ou::joint::simulate_on_regular_grid(
            joint_step,
            joint_step,
            0.0f,
            key,
            0U,
            3U,
            1U,
            observed_states,
            observed_integrated_states
        );
    outputs[12] = observed_states[0];
    outputs[13] = observed_integrated_states[0];
    outputs[14] = grid_terminal.state;
    outputs[15] = grid_terminal.state_integral;

    constexpr float payment_times[] = {2.0f, 3.0f, 4.0f};
    constexpr float accrual_periods[] = {1.0f, 1.0f, 1.0f};
    constexpr std::size_t payment_count = 3U;
    outputs[16] = fitted::forward_rate(
        prepared,
        state,
        0.0f,
        1.0f,
        1.5f,
        0.5f
    );
    outputs[17] = (
        curve::nelson_siegel::discount_factor(initial_curve, 1.0f)
        / curve::nelson_siegel::discount_factor(initial_curve, 1.5f)
        - 1.0f
    ) / 0.5f;
    outputs[18] = fitted::swap_rate(
        prepared,
        state,
        0.0f,
        1.0f,
        payment_times,
        accrual_periods,
        payment_count
    );
    const float swap_annuity =
        curve::nelson_siegel::discount_factor(initial_curve, 2.0f)
        + curve::nelson_siegel::discount_factor(initial_curve, 3.0f)
        + curve::nelson_siegel::discount_factor(initial_curve, 4.0f);
    outputs[19] = (
        curve::nelson_siegel::discount_factor(initial_curve, 1.0f)
        - curve::nelson_siegel::discount_factor(initial_curve, 4.0f)
    ) / swap_annuity;

    constexpr float bond_option_strike = 0.98f;
    outputs[20] = fitted::zero_coupon_bond_call_price(
        prepared,
        0.0f,
        0.0f,
        1.0f,
        1.5f,
        bond_option_strike
    );
    outputs[21] = fitted::zero_coupon_bond_put_price(
        prepared,
        0.0f,
        0.0f,
        1.0f,
        1.5f,
        bond_option_strike
    );
    outputs[22] =
        outputs[20] - outputs[21]
        - (
            curve::nelson_siegel::discount_factor(initial_curve, 1.5f)
            - bond_option_strike
                * curve::nelson_siegel::discount_factor(
                    initial_curve, 1.0f
                )
        );

    const float simple_terminal =
        ou::simulate_terminal_state(terminal_step, 0.0f, 0.5f);
    float simple_observed_states[2] = {};
    const float simple_grid_terminal =
        ou::simulate_on_regular_grid(
            step,
            step,
            0.0f,
            key,
            1U,
            3U,
            1U,
            simple_observed_states
        );
    outputs[23] = simple_terminal;
    outputs[24] = simple_observed_states[0];
    outputs[25] = simple_observed_states[1];
    outputs[26] = simple_grid_terminal;

    const ou::OrnsteinUhlenbeckProcessParameters deterministic = {
        0.15f,
        0.0f,
    };
    const ou::joint::OrnsteinUhlenbeckJointExactTransition deterministic_step =
        ou::joint::prepare_model(deterministic, dt);
    outputs[27] = deterministic_step.integral_state_normal_loading;
    outputs[28] = deterministic_step.integral_independent_standard_deviation;
    outputs[29] = ou::integral_state_loading(0.15f, 1.0f / 252.0f);
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
    require(
        outputs[16] > 0.0f
            && std::fabs(outputs[16] - outputs[17]) < 2.0e-6f,
        "Hull-White forward-rate identity is incorrect"
    );
    require(
        outputs[18] > 0.0f
            && std::fabs(outputs[18] - outputs[19]) < 2.0e-6f,
        "Hull-White swap-rate identity is incorrect"
    );
    require(
        outputs[20] > 0.0f
            && outputs[21] > 0.0f
            && std::fabs(outputs[22]) < 2.0e-6f,
        "Hull-White zero-coupon option parity is incorrect"
    );
    require(
        outputs[27] == 0.0f && outputs[28] == 0.0f,
        "Zero-volatility OU transition contains random loadings"
    );
    require(
        std::fabs(
            outputs[29]
            - static_cast<float>(
                -std::expm1(-0.15 / 252.0) / 0.15
            )
        ) < 1.0e-7f,
        "Small-time OU integral loading is inaccurate"
    );
}
