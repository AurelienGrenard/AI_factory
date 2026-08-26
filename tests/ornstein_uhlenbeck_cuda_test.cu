// Validate exact OU transitions and closed-form fixed-income identities.
#include "common/check_cuda.cuh"

// Include the implementation exactly as future product kernels will.
#include "model/fixed_income/ornstein_uhlenbeck/analytics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

namespace ou =
    ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck;

constexpr std::size_t kOutputCount = 28U;

// Evaluate analytical identities and exact transitions on one CUDA thread.
__global__ void ornstein_uhlenbeck_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    const ou::ModelParameters model = {
        {0.15f, 0.01f},
        0.03f,
    };
    constexpr float state = 0.02f;
    constexpr float valuation_time = 0.5f;
    constexpr float option_expiry = 1.0f;
    constexpr float bond_maturity = 2.0f;
    constexpr float strike = 0.98f;
    constexpr float payment_times[] = {1.0f, 1.5f, 2.0f};
    constexpr float accrual_periods[] = {0.5f, 0.5f, 0.5f};
    constexpr std::uint32_t payment_days[] = {252U, 378U, 504U};
    constexpr float day_fraction = 1.0f / 252.0f;

    outputs[0] = ou::zero_coupon_bond(
        model, state, valuation_time, valuation_time
    );
    outputs[1] = ou::forward_rate(
        model, state, valuation_time, 1.0f, 1.5f, 0.5f
    );
    const float start_bond = ou::zero_coupon_bond(
        model, state, valuation_time, 1.0f
    );
    const float end_bond = ou::zero_coupon_bond(
        model, state, valuation_time, 1.5f
    );
    outputs[2] = (start_bond / end_bond - 1.0f) / 0.5f;
    outputs[3] = ou::swap_rate(
        model,
        state,
        valuation_time,
        0.5f,
        ai_factory::workbench::fixed_income::FixedLegScheduleView{
            payment_times, accrual_periods, 3U,
        }
    );
    const float swap_start_bond = ou::zero_coupon_bond(
        model, state, valuation_time, 0.5f
    );
    const float swap_end_bond = ou::zero_coupon_bond(
        model, state, valuation_time, 2.0f
    );
    const float annuity = 0.5f * (
        ou::zero_coupon_bond(model, state, valuation_time, 1.0f)
        + ou::zero_coupon_bond(model, state, valuation_time, 1.5f)
        + swap_end_bond
    );
    outputs[4] = (swap_start_bond - swap_end_bond) / annuity;

    outputs[5] = ou::zero_coupon_bond_call_price(
        model,
        state,
        valuation_time,
        option_expiry,
        bond_maturity,
        strike
    );
    outputs[6] = ou::zero_coupon_bond_put_price(
        model,
        state,
        valuation_time,
        option_expiry,
        bond_maturity,
        strike
    );
    const float expiry_bond = ou::zero_coupon_bond(
        model, state, valuation_time, option_expiry
    );
    const float underlying_bond = ou::zero_coupon_bond(
        model, state, valuation_time, bond_maturity
    );
    outputs[7] = outputs[5] - outputs[6]
        - (underlying_bond - strike * expiry_bond);

    const ou::ModelParameters deterministic = {
        {0.15f, 0.0f},
        0.03f,
    };
    outputs[8] = ou::zero_coupon_bond(
        deterministic, state, valuation_time, bond_maturity
    );
    outputs[9] = expf(
        -ou::integral_state_loading(0.15f, 1.5f) * state
    );
    outputs[10] = ou::integral_state_loading(1.0e-6f, 1.0e-4f);
    outputs[11] = 1.0e-4f;

    const ou::PreparedModel prepared_model = ou::prepare_model(model.process);
    const ou::PreparedTransition exact =
        ou::prepare_transition(prepared_model, 0.25f);
    float terminal_state = state;
    ou::one_step_transition(exact, 0.75f, terminal_state);
    outputs[12] = terminal_state;
    outputs[13] = fmaf(
        exact.state_decay, state, exact.state_standard_deviation * 0.75f
    );
    const ou::joint::PreparedTransition joint_exact =
        ou::joint::prepare_transition(prepared_model, 0.25f);
    ou::joint::State joint_terminal = {state, 0.0f};
    ou::joint::one_step_transition(
        joint_exact, 0.75f, -0.25f, joint_terminal
    );
    outputs[14] = joint_terminal.state;
    outputs[15] = joint_terminal.state_integral;
    outputs[16] = ou::log_discount_factor(
        model, joint_terminal.state_integral, 0.25f
    );
    outputs[17] = ou::discount_factor(
        model, joint_terminal.state_integral, 0.25f
    );
    const ou::IntegralMoments moments =
        ou::integral_moments(model.process, 0.75f);
    outputs[18] = moments.state_loading
        - ou::integral_state_loading(model.process.mean_reversion, 0.75f);
    outputs[19] = moments.variance
        - ou::integral_variance(model.process, 0.75f);
    outputs[20] = ou::A(model, valuation_time, bond_maturity);
    outputs[21] = ou::B(model, valuation_time, bond_maturity);
    outputs[22] = ou::log_zero_coupon_bond(
        model, state, valuation_time, bond_maturity
    );
    outputs[23] = ou::log_A(model, valuation_time, bond_maturity)
        - outputs[21] * state;
    constexpr float swap_strike = 0.04f;
    outputs[24] = ou::jamshidian_state_boundary(
        model,
        valuation_time,
        swap_strike,
        payment_days,
        accrual_periods,
        day_fraction,
        3U
    );
    outputs[25] =
        swap_strike * 0.5f
            * ou::zero_coupon_bond(
                model, outputs[24], valuation_time, payment_times[0]
            )
        + swap_strike * 0.5f
            * ou::zero_coupon_bond(
                model, outputs[24], valuation_time, payment_times[1]
            )
        + (1.0f + swap_strike * 0.5f)
            * ou::zero_coupon_bond(
                model, outputs[24], valuation_time, payment_times[2]
            );
    outputs[26] = ou::payer_swap_value(
        model,
        state,
        valuation_time,
        valuation_time,
        swap_strike,
        ai_factory::workbench::fixed_income::FixedLegScheduleView{
            payment_times, accrual_periods, 3U,
        }
    );
    outputs[27] = 1.0f - swap_end_bond - swap_strike * annuity;
}

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

// Confirm the reusable OU CUDA primitives over regular and limiting cases.
int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "OU analytics test cudaGetDeviceCount");

    float* device_outputs = nullptr;
    check_cuda(
        cudaMalloc(&device_outputs, kOutputCount * sizeof(float)),
        "OU analytics test cudaMalloc"
    );
    float outputs[kOutputCount] = {};
    try {
        ornstein_uhlenbeck_test_kernel<<<1U, 1U>>>(device_outputs);
        check_cuda(cudaGetLastError(), "OU analytics test kernel launch");
        check_cuda(
            cudaMemcpy(
                outputs,
                device_outputs,
                kOutputCount * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "OU analytics test cudaMemcpy"
        );
        check_cuda(cudaFree(device_outputs), "OU analytics test cudaFree");
        device_outputs = nullptr;
    } catch (...) {
        if (device_outputs != nullptr) cudaFree(device_outputs);
        throw;
    }

    require(outputs[0] == 1.0f, "OU P(t,t) is not one");
    require(
        std::fabs(outputs[1] - outputs[2]) < 2.0e-6f,
        "OU forward-rate identity is incorrect"
    );
    require(
        std::fabs(outputs[3] - outputs[4]) < 2.0e-6f,
        "OU swap-rate identity is incorrect"
    );
    require(
        outputs[5] >= 0.0f && outputs[6] >= 0.0f
            && std::fabs(outputs[7]) < 2.0e-6f,
        "OU zero-coupon option parity is incorrect"
    );
    require(
        std::fabs(outputs[8] - outputs[9]) < 2.0e-6f,
        "OU deterministic zero-coupon is incorrect"
    );
    require(
        std::fabs(outputs[10] - outputs[11]) < 1.0e-10f,
        "OU small-time loading is unstable"
    );
    require(
        std::fabs(outputs[12] - outputs[13]) < 1.0e-7f,
        "OU exact terminal transition is incorrect"
    );
    require(
        std::isfinite(outputs[14]) && std::isfinite(outputs[15])
            && outputs[16] == -outputs[15]
            && std::fabs(outputs[17] - std::exp(outputs[16])) < 1.0e-7f,
        "OU joint transition or path discount is incorrect"
    );
    require(
        std::fabs(outputs[18]) < 1.0e-7f
            && std::fabs(outputs[19]) < 1.0e-9f,
        "OU combined integral moments differ from standalone formulas"
    );
    require(
        outputs[20] > 0.0f && outputs[21] > 0.0f
            && std::fabs(outputs[22] - outputs[23]) < 2.0e-7f,
        "OU affine A/B decomposition is incorrect"
    );
    require(
        std::isfinite(outputs[24])
            && std::fabs(outputs[25] - 1.0f) < 2.0e-6f,
        "OU Jamshidian boundary does not price the coupon bond at par"
    );
    require(
        std::fabs(outputs[26] - outputs[27]) < 2.0e-6f,
        "OU payer swap value is inconsistent with the leg decomposition"
    );
}
