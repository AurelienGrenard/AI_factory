// Verify Bates preparation and the explicit one-step jump composition.
#include "common/check_cuda.cuh"
#include "model/equity/bates/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

struct DynamicsResults {
    ai_factory::workbench::bates::BatesQeParameters prepared;
    ai_factory::workbench::bates::BatesState initial;
    ai_factory::workbench::bates::BatesState no_jump;
    ai_factory::workbench::heston::HestonState heston_no_jump;
    ai_factory::workbench::bates::BatesState with_jumps;
    ai_factory::workbench::heston::HestonState heston_before_jumps;
    ai_factory::workbench::bates::BatesState terminal_first;
    ai_factory::workbench::bates::BatesState terminal_second;
    std::uint32_t poisson_zero;
    std::uint32_t poisson_one;
    std::uint32_t poisson_two;
};

__global__ void exercise_bates_dynamics_kernel(DynamicsResults* output) {
    using namespace ai_factory::workbench;

    constexpr float maturity = 0.5f;
    constexpr std::size_t step_count = 10U;
    const bates::BatesModelParameters parameters = {
        1.25f, 0.03f, 0.01f, 0.05f, 1.4f, 0.04f, 0.32f, -0.65f,
        0.8f, -0.12f, 0.24f,
    };
    const bates::BatesModelParameters no_jump_parameters = {
        1.25f, 0.03f, 0.01f, 0.05f, 1.4f, 0.04f, 0.32f, -0.65f,
        0.0f, -0.12f, 0.24f,
    };
    const bates::BatesQeParameters prepared =
        bates::prepare_model(parameters, maturity, step_count);
    const bates::BatesQeParameters no_jump_prepared =
        bates::prepare_model(no_jump_parameters, maturity, step_count);

    bates::BatesState no_jump = bates::initial_state(no_jump_prepared);
    heston::HestonState heston_no_jump = no_jump;
    constexpr float variance_normal = -0.35f;
    constexpr float variance_uniform = 0.62f;
    constexpr float stock_normal = 0.41f;
    bates::one_step_transition(
        no_jump_prepared,
        variance_normal,
        variance_uniform,
        stock_normal,
        0U,
        0.0f,
        no_jump
    );
    heston::one_step_transition(
        no_jump_prepared.heston,
        variance_normal,
        variance_uniform,
        stock_normal,
        heston_no_jump
    );

    bates::BatesState with_jumps = bates::initial_state(prepared);
    heston::HestonState heston_before_jumps = with_jumps;
    heston::one_step_transition(
        prepared.heston,
        variance_normal,
        variance_uniform,
        stock_normal,
        heston_before_jumps
    );
    bates::one_step_transition(
        prepared,
        variance_normal,
        variance_uniform,
        stock_normal,
        3U,
        -0.27f,
        with_jumps
    );

    const philox::PhiloxKey key = philox::make_key(900000001ULL);
    const bates::BatesState terminal_first =
        bates::simulate_terminal_state(prepared, key, 17U, step_count);
    const bates::BatesState terminal_second =
        bates::simulate_terminal_state(prepared, key, 17U, step_count);
    const std::uint32_t poisson_zero = philox::poisson_from_uniform(
        0.5f, prepared.poisson_mean, prepared.poisson_zero_probability
    );
    const std::uint32_t poisson_one = philox::poisson_from_uniform(
        0.98f, prepared.poisson_mean, prepared.poisson_zero_probability
    );
    const std::uint32_t poisson_two = philox::poisson_from_uniform(
        0.9999f, prepared.poisson_mean, prepared.poisson_zero_probability
    );
    *output = {
        prepared,
        bates::initial_state(prepared),
        no_jump,
        heston_no_jump,
        with_jumps,
        heston_before_jumps,
        terminal_first,
        terminal_second,
        poisson_zero,
        poisson_one,
        poisson_two,
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(float lhs, float rhs, float tolerance = 2.0e-6f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "Bates dynamics test cudaGetDeviceCount");

    DynamicsResults* device_results = nullptr;
    check_cuda(
        cudaMalloc(&device_results, sizeof(DynamicsResults)),
        "Bates dynamics test cudaMalloc"
    );
    exercise_bates_dynamics_kernel<<<1, 1>>>(device_results);
    check_cuda(cudaGetLastError(), "Bates dynamics test kernel launch");
    DynamicsResults results{};
    check_cuda(
        cudaMemcpy(
            &results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "Bates dynamics test cudaMemcpy"
    );
    check_cuda(cudaFree(device_results), "Bates dynamics test cudaFree");

    constexpr float dt = 0.5f / 10.0f;
    const float expected_jump_mean = 0.8f * dt;
    const float expected_compensator = 0.8f * std::expm1(
        -0.12f + 0.5f * 0.24f * 0.24f
    ) * dt;
    require(close(results.prepared.poisson_mean, expected_jump_mean),
            "Bates Poisson mean is incorrect");
    require(close(results.prepared.poisson_zero_probability,
                  std::exp(-expected_jump_mean)),
            "Bates zero-jump probability is incorrect");
    require(close(results.prepared.jump_compensator, expected_compensator),
            "Bates martingale compensator is incorrect");
    require(close(results.initial.log_spot, std::log(1.25f))
                && results.initial.variance == 0.05f,
            "Bates initial state is incorrect");
    require(results.no_jump.log_spot == results.heston_no_jump.log_spot
                && results.no_jump.variance == results.heston_no_jump.variance,
            "Zero-intensity Bates transition differs from Heston");

    const float expected_jump_increment =
        -expected_compensator + 3.0f * -0.12f
        + 0.24f * std::sqrt(3.0f) * -0.27f;
    require(close(results.with_jumps.log_spot
                      - results.heston_before_jumps.log_spot,
                  expected_jump_increment),
            "Bates compound-Poisson log increment is incorrect");
    require(results.with_jumps.variance
                == results.heston_before_jumps.variance,
            "Bates jump update changed the variance state");
    require(results.terminal_first.log_spot == results.terminal_second.log_spot
                && results.terminal_first.variance
                    == results.terminal_second.variance,
            "Bates terminal simulation is not deterministic");
    require(results.poisson_zero == 0U
                && results.poisson_one == 1U
                && results.poisson_two == 2U,
            "Philox Poisson inverse CDF returned incorrect quantiles");
    return 0;
}
