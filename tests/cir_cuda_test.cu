// Validate exact CIR transitions and affine fixed-income identities.
#include "common/check_cuda.cuh"

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

namespace cir = ai_factory::workbench::model::cir;
namespace philox = ai_factory::workbench::philox;

constexpr std::size_t kOutputCount = 16U;

// Evaluate deterministic identities and replay checks on one CUDA thread.
__global__ void cir_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    const cir::CirModelParameters model = {
        {0.60f, 0.04f, 0.15f},
        0.03f,
    };
    constexpr float state = 0.025f;
    constexpr float valuation_time = 0.5f;
    constexpr float maturity = 4.0f;
    constexpr float payment_times[] = {1.0f, 1.5f, 2.0f};
    constexpr float accrual_periods[] = {0.5f, 0.5f, 0.5f};

    outputs[0] = cir::zero_coupon_bond(
        model, state, valuation_time, valuation_time
    );
    outputs[1] = cir::A(model, valuation_time, maturity);
    outputs[2] = cir::B(model, valuation_time, maturity);
    outputs[3] = cir::zero_coupon_bond(
        model, state, valuation_time, maturity
    );
    outputs[4] = outputs[1] * expf(-outputs[2] * state);
    outputs[5] = cir::log_zero_coupon_bond(
        model, state, valuation_time, maturity
    );
    outputs[6] = cir::forward_rate(
        model, state, valuation_time, 1.0f, 1.5f, 0.5f
    );
    const float start_bond = cir::zero_coupon_bond(
        model, state, valuation_time, 1.0f
    );
    const float end_bond = cir::zero_coupon_bond(
        model, state, valuation_time, 1.5f
    );
    outputs[7] = (start_bond / end_bond - 1.0f) / 0.5f;
    outputs[8] = cir::swap_rate(
        model,
        state,
        valuation_time,
        0.5f,
        payment_times,
        accrual_periods,
        3U
    );
    const float swap_start_bond = cir::zero_coupon_bond(
        model, state, valuation_time, 0.5f
    );
    const float swap_end_bond = cir::zero_coupon_bond(
        model, state, valuation_time, 2.0f
    );
    const float annuity = 0.5f * (
        cir::zero_coupon_bond(model, state, valuation_time, 1.0f)
        + cir::zero_coupon_bond(model, state, valuation_time, 1.5f)
        + swap_end_bond
    );
    outputs[9] = (swap_start_bond - swap_end_bond) / annuity;

    const cir::CirExactTransition short_transition = cir::prepare_model(
        model.process, 1.0e-4f
    );
    constexpr std::size_t path = 73U;
    const philox::PhiloxKey key = philox::make_key(900000301ULL);
    outputs[10] = cir::simulate_terminal_state(
        short_transition, state, key, path
    );
    outputs[11] = cir::simulate_terminal_state(
        short_transition, state, key, path
    );
    const cir::CirExactTransition regular_transition = cir::prepare_model(
        model.process, 0.25f
    );
    outputs[12] = cir::simulate_terminal_state(
        regular_transition, 0.0f, key, path
    );
    outputs[13] = regular_transition.degrees_of_freedom;
    outputs[14] = cir::log_discount_factor(0.25f);
    outputs[15] = cir::discount_factor(0.25f);
}

// Draw one exact endpoint per independent Philox path for moment checks.
__global__ void sample_cir_kernel(
    cir::CirExactTransition transition,
    float initial_state,
    std::uint64_t seed,
    std::size_t sample_count,
    float* samples
) {
    const std::size_t path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= sample_count) return;
    samples[path] = cir::simulate_terminal_state(
        transition,
        initial_state,
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

    constexpr std::size_t sample_count = 1U << 18U;
    constexpr unsigned int threads_per_block = 256U;
    constexpr unsigned int block_count = static_cast<unsigned int>(
        (sample_count + threads_per_block - 1U) / threads_per_block
    );
    constexpr cir::CirProcessParameters process = {0.60f, 0.04f, 0.15f};
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
        cir::CirExactTransition host_transition = {
            static_cast<float>(decay),
            4.0f * process.mean_reversion * process.long_term_mean
                / (process.volatility * process.volatility),
            process.volatility * process.volatility
                * static_cast<float>(one_minus_decay)
                / (4.0f * process.mean_reversion),
        };
        sample_cir_kernel<<<block_count, threads_per_block>>>(
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
    return 0;
}
