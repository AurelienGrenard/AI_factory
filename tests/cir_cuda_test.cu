// Validate exact CIR transitions and affine fixed-income identities.
#include "common/check_cuda.cuh"
#include "common/simulation/path_simulation.cuh"

// Include the implementation exactly as future product kernels will.
#include "model/fixed_income/cir/analytics.cu"
#include "model/fixed_income/cir/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

namespace cir = ai_factory::workbench::model::fixed_income::cir;
namespace philox = ai_factory::workbench::philox;

constexpr std::size_t kOutputCount = 22U;

// Evaluate deterministic identities and replay checks on one CUDA thread.
__global__ void cir_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    const ai_factory::workbench::model::fixed_income::cir::ModelParameters model = {
        {0.60f, 0.04f, 0.15f},
        0.03f,
    };
    constexpr float state = 0.025f;
    constexpr float valuation_time = 0.5f;
    constexpr float maturity = 4.0f;
    constexpr float payment_times[] = {1.0f, 1.5f, 2.0f};
    constexpr float accrual_periods[] = {0.5f, 0.5f, 0.5f};

    outputs[0] = ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(
        model, state, valuation_time, valuation_time
    );
    outputs[1] = ai_factory::workbench::model::fixed_income::cir::A(model, valuation_time, maturity);
    outputs[2] = ai_factory::workbench::model::fixed_income::cir::B(model, valuation_time, maturity);
    outputs[3] = ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(
        model, state, valuation_time, maturity
    );
    outputs[4] = outputs[1] * expf(-outputs[2] * state);
    outputs[5] = ai_factory::workbench::model::fixed_income::cir::log_zero_coupon_bond(
        model, state, valuation_time, maturity
    );
    outputs[6] = ai_factory::workbench::model::fixed_income::cir::forward_rate(
        model, state, valuation_time, 1.0f, 1.5f, 0.5f
    );
    const float start_bond = ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(
        model, state, valuation_time, 1.0f
    );
    const float end_bond = ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(
        model, state, valuation_time, 1.5f
    );
    outputs[7] = (start_bond / end_bond - 1.0f) / 0.5f;
    outputs[8] = ai_factory::workbench::model::fixed_income::cir::swap_rate(
        model,
        state,
        valuation_time,
        0.5f,
        ai_factory::workbench::fixed_income::FixedLegScheduleView{
            payment_times, accrual_periods, 3U,
        }
    );
    const float swap_start_bond = ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(
        model, state, valuation_time, 0.5f
    );
    const float swap_end_bond = ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(
        model, state, valuation_time, 2.0f
    );
    const float annuity = 0.5f * (
        ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(model, state, valuation_time, 1.0f)
        + ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(model, state, valuation_time, 1.5f)
        + swap_end_bond
    );
    outputs[9] = (swap_start_bond - swap_end_bond) / annuity;

    ai_factory::workbench::model::fixed_income::cir::PreparedModel prepared_model = ai_factory::workbench::model::fixed_income::cir::prepare_model(model.process);
    prepared_model.initial_state = state;
    const ai_factory::workbench::model::fixed_income::cir::PreparedTransition short_transition =
        ai_factory::workbench::model::fixed_income::cir::prepare_transition(prepared_model, 1.0e-4f);
    constexpr std::size_t path = 73U;
    const philox::PhiloxKey key = philox::make_key(900000301ULL);
    outputs[10] =
        ai_factory::workbench::simulation::
            simulate_exact_transition_terminal<ai_factory::workbench::model::fixed_income::cir::DynamicsPolicy>(
                prepared_model, short_transition, key, path
            );
    outputs[11] =
        ai_factory::workbench::simulation::
            simulate_exact_transition_terminal<ai_factory::workbench::model::fixed_income::cir::DynamicsPolicy>(
                prepared_model, short_transition, key, path
            );
    const ai_factory::workbench::model::fixed_income::cir::PreparedTransition regular_transition =
        ai_factory::workbench::model::fixed_income::cir::prepare_transition(prepared_model, 0.25f);
    ai_factory::workbench::model::fixed_income::cir::PreparedModel zero_initial_model = prepared_model;
    zero_initial_model.initial_state = 0.0f;
    outputs[12] =
        ai_factory::workbench::simulation::
            simulate_exact_transition_terminal<ai_factory::workbench::model::fixed_income::cir::DynamicsPolicy>(
                zero_initial_model, regular_transition, key, path
            );
    outputs[13] = prepared_model.degrees_of_freedom;
    outputs[14] = ai_factory::workbench::model::fixed_income::cir::log_discount_factor(
        model, 0.25f, 0.25f
    );
    outputs[15] = ai_factory::workbench::model::fixed_income::cir::discount_factor(
        model, 0.25f, 0.25f
    );

    constexpr float joint_delta_t = 1.0f / 64.0f;
    constexpr std::uint32_t joint_step_count = 64U;
    outputs[16] = ai_factory::workbench::model::fixed_income::cir::zero_coupon_bond(
        model, model.initial_state, 0.0f, 1.0f
    );
    const ai_factory::workbench::model::fixed_income::cir::PreparedDynamics state_dynamics =
        ai_factory::workbench::model::fixed_income::cir::DynamicsPolicy::prepare_dynamics(model, joint_delta_t);
    const ai_factory::workbench::model::fixed_income::cir::joint::PreparedDynamics joint_dynamics =
        ai_factory::workbench::model::fixed_income::cir::joint::DynamicsPolicy::prepare_dynamics(model, joint_delta_t);
    outputs[17] =
        ai_factory::workbench::simulation::simulate_fixed_step_terminal<
            ai_factory::workbench::model::fixed_income::cir::DynamicsPolicy
        >(state_dynamics, joint_step_count, key, path);
    const ai_factory::workbench::model::fixed_income::cir::joint::State joint_terminal =
        ai_factory::workbench::simulation::simulate_fixed_step_terminal<
            ai_factory::workbench::model::fixed_income::cir::joint::DynamicsPolicy
        >(joint_dynamics, joint_step_count, key, path);
    outputs[18] = joint_terminal.state;
    outputs[19] = joint_terminal.state_integral;
    const ai_factory::workbench::model::fixed_income::cir::joint::State one_step_joint =
        ai_factory::workbench::simulation::simulate_fixed_step_terminal<
            ai_factory::workbench::model::fixed_income::cir::joint::DynamicsPolicy
        >(joint_dynamics, 1U, key, path + 1U);
    outputs[20] = one_step_joint.state_integral;
    outputs[21] = fmaf(
        0.5f * joint_delta_t,
        model.initial_state + one_step_joint.state,
        0.0f
    );
}

// Draw one exact endpoint per independent Philox path for moment checks.
__global__ void sample_cir_kernel(
    ai_factory::workbench::model::fixed_income::cir::PreparedModel model,
    ai_factory::workbench::model::fixed_income::cir::PreparedTransition transition,
    float initial_state,
    std::uint64_t seed,
    std::size_t sample_count,
    float* samples
) {
    const std::size_t path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= sample_count) return;
    model.initial_state = initial_state;
    samples[path] =
        ai_factory::workbench::simulation::
            simulate_exact_transition_terminal<ai_factory::workbench::model::fixed_income::cir::DynamicsPolicy>(
                model, transition, philox::make_key(seed), path
            );
}

// Draw discretized joint rate/integral paths for analytical moment checks.
__global__ void sample_joint_cir_kernel(
    ai_factory::workbench::model::fixed_income::cir::ModelParameters parameters,
    float delta_t,
    std::uint32_t step_count,
    std::uint64_t seed,
    std::size_t sample_count,
    ai_factory::workbench::model::fixed_income::cir::joint::State* samples
) {
    const std::size_t path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= sample_count) return;
    const ai_factory::workbench::model::fixed_income::cir::joint::PreparedDynamics dynamics =
        ai_factory::workbench::model::fixed_income::cir::joint::DynamicsPolicy::prepare_dynamics(parameters, delta_t);
    samples[path] =
        ai_factory::workbench::simulation::simulate_fixed_step_terminal<
            ai_factory::workbench::model::fixed_income::cir::joint::DynamicsPolicy
        >(
            dynamics,
            step_count,
            philox::make_key(seed),
            path
        );
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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
    check_cuda(availability, "CIR test cudaGetDeviceCount");

    float* device_outputs = nullptr;
    check_cuda(
        cudaMalloc(&device_outputs, kOutputCount * sizeof(float)),
        "CIR test output cudaMalloc"
    );
    float outputs[kOutputCount] = {};
    try {
        cir_test_kernel<<<1U, 1U>>>(device_outputs);
        check_cuda(cudaGetLastError(), "CIR identity test kernel launch");
        check_cuda(
            cudaMemcpy(
                outputs,
                device_outputs,
                sizeof(outputs),
                cudaMemcpyDeviceToHost
            ),
            "CIR identity test cudaMemcpy"
        );
        check_cuda(cudaFree(device_outputs), "CIR test output cudaFree");
        device_outputs = nullptr;
    } catch (...) {
        if (device_outputs != nullptr) cudaFree(device_outputs);
        throw;
    }

    require(outputs[0] == 1.0f, "CIR P(t,t) is not one");
    require(
        outputs[1] > 0.0f && outputs[2] > 0.0f
            && std::fabs(outputs[3] - outputs[4]) < 2.0e-7f
            && std::fabs(outputs[5] - std::log(outputs[3])) < 2.0e-7f,
        "CIR affine A/B zero-coupon identity is incorrect"
    );
    require(
        std::fabs(outputs[6] - outputs[7]) < 2.0e-6f,
        "CIR forward-rate identity is incorrect"
    );
    require(
        std::fabs(outputs[8] - outputs[9]) < 2.0e-6f,
        "CIR swap-rate identity is incorrect"
    );
    require(
        outputs[10] >= 0.0f && outputs[10] == outputs[11],
        "CIR large-intensity exact transition is not replayable"
    );
    require(
        outputs[12] >= 0.0f && outputs[13] > 0.0f,
        "CIR exact transition does not preserve non-negativity"
    );
    require(
        outputs[14] == -0.25f
            && std::fabs(outputs[15] - std::exp(-0.25f)) < 1.0e-7f,
        "CIR path discount-factor identity is incorrect"
    );
    require(
        outputs[17] == outputs[18] && outputs[19] >= 0.0f,
        "CIR joint dynamics changed the exact endpoint transition"
    );
    require(
        outputs[20] == outputs[21],
        "CIR joint dynamics does not use trapezoidal integration"
    );

    constexpr std::size_t sample_count = 1U << 18U;
    constexpr unsigned int threads_per_block = 256U;
    constexpr unsigned int block_count = static_cast<unsigned int>(
        (sample_count + threads_per_block - 1U) / threads_per_block
    );
    constexpr ai_factory::workbench::model::fixed_income::cir::ProcessParameters process = {0.60f, 0.04f, 0.15f};
    constexpr float initial_state = 0.03f;
    constexpr float time_interval = 0.75f;

    // Prepare the same coefficients in double precision for reference moments.
    const double decay = std::exp(
        -static_cast<double>(process.mean_reversion) * time_interval
    );
    const double sigma_squared =
        static_cast<double>(process.volatility) * process.volatility;
    const double one_minus_decay = 1.0 - decay;
    const double expected_mean = process.long_term_mean
        + (initial_state - process.long_term_mean) * decay;
    const double expected_variance =
        initial_state * sigma_squared * decay * one_minus_decay
            / process.mean_reversion
        + process.long_term_mean * sigma_squared
            * one_minus_decay * one_minus_decay
            / (2.0 * process.mean_reversion);

    float* device_samples = nullptr;
    check_cuda(
        cudaMalloc(&device_samples, sample_count * sizeof(float)),
        "CIR moment test cudaMalloc"
    );
    std::vector<float> samples(sample_count);
    try {
        // The prepared values are deterministic functions of the row and dt.
        const ai_factory::workbench::model::fixed_income::cir::PreparedModel host_model = {
            process.mean_reversion,
            4.0f * process.mean_reversion * process.long_term_mean
                / (process.volatility * process.volatility),
            process.volatility * process.volatility
                / (4.0f * process.mean_reversion),
        };
        const ai_factory::workbench::model::fixed_income::cir::PreparedTransition host_transition = {
            static_cast<float>(decay),
            process.volatility * process.volatility
                * static_cast<float>(one_minus_decay)
                / (4.0f * process.mean_reversion),
        };
        sample_cir_kernel<<<block_count, threads_per_block>>>(
            host_model,
            host_transition,
            initial_state,
            900000401ULL,
            sample_count,
            device_samples
        );
        check_cuda(cudaGetLastError(), "CIR moment test kernel launch");
        check_cuda(
            cudaMemcpy(
                samples.data(),
                device_samples,
                sample_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "CIR moment test cudaMemcpy"
        );
        check_cuda(cudaFree(device_samples), "CIR moment test cudaFree");
        device_samples = nullptr;
    } catch (...) {
        if (device_samples != nullptr) cudaFree(device_samples);
        throw;
    }

    long double sum = 0.0L;
    long double squared_sum = 0.0L;
    for (const float sample : samples) {
        require(
            std::isfinite(sample) && sample >= 0.0f,
            "CIR exact transition produced an invalid state"
        );
        const long double value = sample;
        sum += value;
        squared_sum += value * value;
    }
    const long double count = static_cast<long double>(sample_count);
    const double empirical_mean = static_cast<double>(sum / count);
    const double empirical_variance = static_cast<double>(
        squared_sum / count - (sum / count) * (sum / count)
    );
    require(
        std::fabs(empirical_mean - expected_mean) < 7.5e-5,
        "CIR exact transition mean is outside tolerance"
    );
    require(
        std::fabs(empirical_variance - expected_variance) < 1.5e-5,
        "CIR exact transition variance is outside tolerance"
    );

    constexpr std::size_t joint_sample_count = 1U << 15U;
    constexpr std::uint32_t joint_step_count = 64U;
    constexpr float joint_delta_t = 1.0f / 64.0f;
    constexpr unsigned int joint_block_count = static_cast<unsigned int>(
        (joint_sample_count + threads_per_block - 1U) / threads_per_block
    );
    ai_factory::workbench::model::fixed_income::cir::joint::State* device_joint_samples = nullptr;
    check_cuda(
        cudaMalloc(
            &device_joint_samples,
            joint_sample_count * sizeof(ai_factory::workbench::model::fixed_income::cir::joint::State)
        ),
        "CIR joint moment test cudaMalloc"
    );
    std::vector<ai_factory::workbench::model::fixed_income::cir::joint::State> joint_samples(joint_sample_count);
    try {
        sample_joint_cir_kernel<<<joint_block_count, threads_per_block>>>(
            ai_factory::workbench::model::fixed_income::cir::ModelParameters{process, initial_state},
            joint_delta_t,
            joint_step_count,
            900000501ULL,
            joint_sample_count,
            device_joint_samples
        );
        check_cuda(
            cudaGetLastError(),
            "CIR joint moment test kernel launch"
        );
        check_cuda(
            cudaMemcpy(
                joint_samples.data(),
                device_joint_samples,
                joint_sample_count * sizeof(ai_factory::workbench::model::fixed_income::cir::joint::State),
                cudaMemcpyDeviceToHost
            ),
            "CIR joint moment test cudaMemcpy"
        );
        check_cuda(
            cudaFree(device_joint_samples),
            "CIR joint moment test cudaFree"
        );
        device_joint_samples = nullptr;
    } catch (...) {
        if (device_joint_samples != nullptr) cudaFree(device_joint_samples);
        throw;
    }

    long double integral_sum = 0.0L;
    long double discount_sum = 0.0L;
    for (const ai_factory::workbench::model::fixed_income::cir::joint::State& sample : joint_samples) {
        require(
            std::isfinite(sample.state) && sample.state >= 0.0f
                && std::isfinite(sample.state_integral)
                && sample.state_integral >= 0.0f,
            "CIR joint dynamics produced an invalid state"
        );
        integral_sum += sample.state_integral;
        discount_sum += std::exp(-static_cast<long double>(
            sample.state_integral
        ));
    }
    const long double joint_count =
        static_cast<long double>(joint_sample_count);
    const double empirical_integral_mean = static_cast<double>(
        integral_sum / joint_count
    );
    const double empirical_discount = static_cast<double>(
        discount_sum / joint_count
    );
    const double expected_integral_mean = process.long_term_mean
        + (initial_state - process.long_term_mean)
            * (1.0 - std::exp(-static_cast<double>(
                process.mean_reversion
            )))
            / process.mean_reversion;
    require(
        std::fabs(empirical_integral_mean - expected_integral_mean) < 3.0e-4,
        "CIR integrated-rate mean is outside tolerance"
    );
    require(
        std::fabs(empirical_discount - outputs[16]) < 3.0e-4,
        "CIR path discount factor is outside the affine bond tolerance"
    );
    return 0;
}
