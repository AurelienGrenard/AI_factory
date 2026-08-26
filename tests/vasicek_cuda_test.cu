// Validate exact Vasicek transitions and closed-form fixed-income identities.
#include "common/check_cuda.cuh"

// Include the implementation exactly as future product kernels will.
#include "model/fixed_income/vasicek/analytics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

namespace vasicek =
    ai_factory::workbench::model::fixed_income::vasicek;

constexpr std::size_t kOutputCount = 25U;

// Evaluate analytical identities and exact transitions on one CUDA thread.
__global__ void vasicek_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    const ai_factory::workbench::model::fixed_income::vasicek::ModelParameters model = {
        {0.15f, 0.04f, 0.01f},
        0.03f,
    };
    constexpr float state = 0.02f;
    constexpr float valuation_time = 0.5f;
    constexpr float option_expiry = 1.0f;
    constexpr float bond_maturity = 2.0f;
    constexpr float strike = 0.98f;
    constexpr float payment_times[] = {1.0f, 1.5f, 2.0f};
    constexpr float accrual_periods[] = {0.5f, 0.5f, 0.5f};

    outputs[0] = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(
        model, state, valuation_time, valuation_time
    );
    outputs[1] = ai_factory::workbench::model::fixed_income::vasicek::forward_rate(
        model, state, valuation_time, 1.0f, 1.5f, 0.5f
    );
    const float start_bond = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(
        model, state, valuation_time, 1.0f
    );
    const float end_bond = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(
        model, state, valuation_time, 1.5f
    );
    outputs[2] = (start_bond / end_bond - 1.0f) / 0.5f;
    outputs[3] = ai_factory::workbench::model::fixed_income::vasicek::swap_rate(
        model,
        state,
        valuation_time,
        0.5f,
        ai_factory::workbench::fixed_income::FixedLegScheduleView{
            payment_times, accrual_periods, 3U,
        }
    );
    const float swap_start_bond = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(
        model, state, valuation_time, 0.5f
    );
    const float swap_end_bond = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(
        model, state, valuation_time, 2.0f
    );
    const float annuity = 0.5f * (
        ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(model, state, valuation_time, 1.0f)
        + ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(model, state, valuation_time, 1.5f)
        + swap_end_bond
    );
    outputs[4] = (swap_start_bond - swap_end_bond) / annuity;

    outputs[5] = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond_call_price(
        model,
        state,
        valuation_time,
        option_expiry,
        bond_maturity,
        strike
    );
    outputs[6] = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond_put_price(
        model,
        state,
        valuation_time,
        option_expiry,
        bond_maturity,
        strike
    );
    const float expiry_bond = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(
        model, state, valuation_time, option_expiry
    );
    const float underlying_bond = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(
        model, state, valuation_time, bond_maturity
    );
    outputs[7] = outputs[5] - outputs[6]
        - (underlying_bond - strike * expiry_bond);

    const ai_factory::workbench::model::fixed_income::vasicek::ModelParameters deterministic = {
        {0.15f, 0.04f, 0.0f},
        0.03f,
    };
    outputs[8] = ai_factory::workbench::model::fixed_income::vasicek::zero_coupon_bond(
        deterministic, state, valuation_time, bond_maturity
    );
    const float deterministic_loading =
        ai_factory::workbench::model::fixed_income::vasicek::integral_state_loading(0.15f, 1.5f);
    outputs[9] = expf(
        -deterministic_loading * state
        - deterministic.process.long_term_mean
            * (1.5f - deterministic_loading)
    );
    outputs[10] = ai_factory::workbench::model::fixed_income::vasicek::integral_state_loading(1.0e-6f, 1.0e-4f);
    outputs[11] = 1.0e-4f;

    const ai_factory::workbench::model::fixed_income::vasicek::PreparedModel prepared_model =
        ai_factory::workbench::model::fixed_income::vasicek::prepare_model(model.process);
    const ai_factory::workbench::model::fixed_income::vasicek::PreparedTransition exact =
        ai_factory::workbench::model::fixed_income::vasicek::prepare_transition(prepared_model, 0.25f);
    float terminal_state = state;
    ai_factory::workbench::model::fixed_income::vasicek::one_step_transition(exact, 0.75f, terminal_state);
    outputs[12] = terminal_state;
    outputs[13] = fmaf(
        exact.state_decay,
        state,
        exact.state_mean_increment
            + exact.state_standard_deviation * 0.75f
    );
    const ai_factory::workbench::model::fixed_income::vasicek::joint::PreparedTransition joint_exact =
        ai_factory::workbench::model::fixed_income::vasicek::joint::prepare_transition(prepared_model, 0.25f);
    ai_factory::workbench::model::fixed_income::vasicek::joint::State joint_terminal = {state, 0.0f};
    ai_factory::workbench::model::fixed_income::vasicek::joint::one_step_transition(
        joint_exact, 0.75f, -0.25f, joint_terminal
    );
    outputs[14] = joint_terminal.state;
    outputs[15] = joint_terminal.state_integral;
    outputs[16] = ai_factory::workbench::model::fixed_income::vasicek::log_discount_factor(
        model, joint_terminal.state_integral, 0.25f
    );
    outputs[17] = ai_factory::workbench::model::fixed_income::vasicek::discount_factor(
        model, joint_terminal.state_integral, 0.25f
    );
    const ai_factory::workbench::model::fixed_income::vasicek::IntegralMoments moments =
        ai_factory::workbench::model::fixed_income::vasicek::integral_moments(model.process, 0.75f);
    outputs[18] = moments.state_loading
        - ai_factory::workbench::model::fixed_income::vasicek::integral_state_loading(model.process.mean_reversion, 0.75f);
    outputs[19] = moments.variance
        - ai_factory::workbench::model::fixed_income::vasicek::integral_variance(model.process, 0.75f);
    outputs[20] = moments.state_mean_increment
        - model.process.long_term_mean * (0.75f - moments.state_loading);
    outputs[21] = ai_factory::workbench::model::fixed_income::vasicek::A(model, valuation_time, bond_maturity);
    outputs[22] = ai_factory::workbench::model::fixed_income::vasicek::B(model, valuation_time, bond_maturity);
    outputs[23] = ai_factory::workbench::model::fixed_income::vasicek::log_zero_coupon_bond(
        model, state, valuation_time, bond_maturity
    );
    outputs[24] = ai_factory::workbench::model::fixed_income::vasicek::log_A(
        model, valuation_time, bond_maturity
    ) - outputs[22] * state;
}

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

// Confirm the reusable Vasicek CUDA primitives over regular and limiting cases.
int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "Vasicek analytics test cudaGetDeviceCount");

    float* device_outputs = nullptr;
    check_cuda(
        cudaMalloc(&device_outputs, kOutputCount * sizeof(float)),
        "Vasicek analytics test cudaMalloc"
    );
    float outputs[kOutputCount] = {};
    try {
        vasicek_test_kernel<<<1U, 1U>>>(device_outputs);
        check_cuda(cudaGetLastError(), "Vasicek analytics test kernel launch");
        check_cuda(
            cudaMemcpy(
                outputs,
                device_outputs,
                kOutputCount * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Vasicek analytics test cudaMemcpy"
        );
        check_cuda(cudaFree(device_outputs), "Vasicek analytics test cudaFree");
        device_outputs = nullptr;
    } catch (...) {
        if (device_outputs != nullptr) cudaFree(device_outputs);
        throw;
    }

    require(outputs[0] == 1.0f, "Vasicek P(t,t) is not one");
    require(
        std::fabs(outputs[1] - outputs[2]) < 2.0e-6f,
        "Vasicek forward-rate identity is incorrect"
    );
    require(
        std::fabs(outputs[3] - outputs[4]) < 2.0e-6f,
        "Vasicek swap-rate identity is incorrect"
    );
    require(
        outputs[5] >= 0.0f && outputs[6] >= 0.0f
            && std::fabs(outputs[7]) < 2.0e-6f,
        "Vasicek zero-coupon option parity is incorrect"
    );
    require(
        std::fabs(outputs[8] - outputs[9]) < 2.0e-6f,
        "Vasicek deterministic zero-coupon is incorrect"
    );
    require(
        std::fabs(outputs[10] - outputs[11]) < 1.0e-10f,
        "Vasicek small-time loading is unstable"
    );
    require(
        std::fabs(outputs[12] - outputs[13]) < 1.0e-7f,
        "Vasicek exact terminal transition is incorrect"
    );
    require(
        std::isfinite(outputs[14]) && std::isfinite(outputs[15])
            && outputs[16] == -outputs[15]
            && std::fabs(outputs[17] - std::exp(outputs[16])) < 1.0e-7f,
        "Vasicek joint transition or path discount is incorrect"
    );
    require(
        std::fabs(outputs[18]) < 1.0e-7f
            && std::fabs(outputs[19]) < 1.0e-9f
            && std::fabs(outputs[20]) < 1.0e-8f,
        "Vasicek combined integral moments differ from standalone formulas"
    );
    require(
        outputs[21] > 0.0f && outputs[22] > 0.0f
            && std::fabs(outputs[23] - outputs[24]) < 2.0e-7f,
        "Vasicek affine A/B decomposition is incorrect"
    );
}
