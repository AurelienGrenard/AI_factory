// Verify the common samplers and exact VG/NIG dynamics contracts.
#include "common/check_cuda.cuh"
#include "common/equity/observation_handlers.cuh"
#include "common/equity/path_simulation.cuh"
#include "model/equity/normal_inverse_gaussian/dynamics.cu"
#include "model/equity/variance_gamma/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

struct LevyDynamicsResults {
    ai_factory::workbench::variance_gamma::PreparedModel vg;
    ai_factory::workbench::variance_gamma::PreparedTransition
        vg_transition_coefficients;
    ai_factory::workbench::variance_gamma::PreparedTransition
        vg_exact;
    ai_factory::workbench::normal_inverse_gaussian::
        PreparedModel nig;
    ai_factory::workbench::normal_inverse_gaussian::
        PreparedTransition nig_transition_coefficients;
    ai_factory::workbench::normal_inverse_gaussian::
        PreparedTransition nig_exact;
    ai_factory::workbench::variance_gamma::State vg_transition;
    ai_factory::workbench::normal_inverse_gaussian::
        State nig_transition;
    float gamma_small_shape_first;
    float gamma_small_shape_replay;
    float gamma_large_shape_first;
    float gamma_large_shape_replay;
    float inverse_gaussian_first;
    float inverse_gaussian_replay;
    float vg_terminal_first;
    float vg_terminal_replay;
    float vg_terminal_manual_aggregate;
    float nig_terminal_first;
    float nig_terminal_replay;
    float nig_terminal_manual_aggregate;
    std::uint32_t exact_regular_matches_calendar;
};

__global__ void exercise_levy_dynamics_kernel(LevyDynamicsResults* output) {
    using namespace ai_factory::workbench;

    constexpr float maturity = 0.5f;
    constexpr std::size_t step_count = 10U;
    constexpr float delta_t = maturity / static_cast<float>(step_count);
    const variance_gamma::ModelParameters vg_parameters = {
        100.0f, 0.03f, 0.01f, 0.2f, 0.25f, -0.1f,
    };
    const normal_inverse_gaussian::ModelParameters
        nig_parameters = {
            100.0f, 0.03f, 0.01f, 8.0f, -2.0f, 0.4f,
        };
    const auto vg = variance_gamma::prepare_model(vg_parameters);
    const auto vg_transition_coefficients =
        variance_gamma::prepare_transition(vg, delta_t);
    const auto vg_exact =
        variance_gamma::prepare_transition(vg, maturity);
    const auto nig = normal_inverse_gaussian::prepare_model(nig_parameters);
    const auto nig_transition_coefficients =
        normal_inverse_gaussian::prepare_transition(nig, delta_t);
    const auto nig_exact =
        normal_inverse_gaussian::prepare_transition(nig, maturity);

    auto vg_transition = variance_gamma::initial_state(vg);
    variance_gamma::one_step_transition(
        vg, vg_transition_coefficients, 0.08f, -0.4f, vg_transition
    );
    auto nig_transition = normal_inverse_gaussian::initial_state(nig);
    normal_inverse_gaussian::one_step_transition(
        nig, nig_transition_coefficients, 0.03f, 0.5f, nig_transition
    );

    const philox::PhiloxKey key = philox::make_key(900000001ULL);
    philox::UniformSequence first_sampler_uniforms(key, 17ULL);
    philox::NormalPairCache first_sampler_cache;
    const float gamma_small_shape_first = philox::marsaglia_tsang_gamma(
        first_sampler_uniforms, first_sampler_cache, 0.4f, 0.7f
    );
    const float gamma_large_shape_first = philox::marsaglia_tsang_gamma(
        first_sampler_uniforms, first_sampler_cache, 2.5f, 0.8f
    );
    const float inverse_gaussian_first =
        philox::michael_schucany_haas_inverse_gaussian(
            first_sampler_uniforms, first_sampler_cache, 0.8f, 1.7f
        );

    philox::UniformSequence replay_sampler_uniforms(key, 17ULL);
    philox::NormalPairCache replay_sampler_cache;
    const float gamma_small_shape_replay = philox::marsaglia_tsang_gamma(
        replay_sampler_uniforms, replay_sampler_cache, 0.4f, 0.7f
    );
    const float gamma_large_shape_replay = philox::marsaglia_tsang_gamma(
        replay_sampler_uniforms, replay_sampler_cache, 2.5f, 0.8f
    );
    const float inverse_gaussian_replay =
        philox::michael_schucany_haas_inverse_gaussian(
            replay_sampler_uniforms, replay_sampler_cache, 0.8f, 1.7f
        );

    const float vg_terminal_first =
        equity::simulate_exact_transition_terminal<
            variance_gamma::DynamicsPolicy
        >(vg, vg_exact, key, 23U).log_spot;
    const float vg_terminal_replay =
        equity::simulate_exact_transition_terminal<
            variance_gamma::DynamicsPolicy
        >(vg, vg_exact, key, 23U).log_spot;

    auto vg_terminal_manual = variance_gamma::initial_state(vg);
    philox::UniformSequence vg_uniforms(key, 23ULL);
    philox::NormalPairCache vg_cache;
    const float vg_gamma = philox::marsaglia_tsang_gamma(
        vg_uniforms,
        vg_cache,
        vg_exact.gamma_shape,
        vg.nu
    );
    const float vg_normal = philox::next_normal(vg_uniforms, vg_cache);
    variance_gamma::one_step_transition(
        vg,
        vg_exact,
        vg_gamma,
        vg_normal,
        vg_terminal_manual
    );

    const float nig_terminal_first =
        equity::simulate_exact_transition_terminal<
            normal_inverse_gaussian::DynamicsPolicy
        >(
            nig, nig_exact, key, 29U
        ).log_spot;
    const float nig_terminal_replay =
        equity::simulate_exact_transition_terminal<
            normal_inverse_gaussian::DynamicsPolicy
        >(
            nig, nig_exact, key, 29U
        ).log_spot;

    auto nig_terminal_manual = normal_inverse_gaussian::initial_state(
        nig
    );
    philox::UniformSequence nig_uniforms(key, 29ULL);
    philox::NormalPairCache nig_cache;
    const float nig_clock =
        philox::michael_schucany_haas_inverse_gaussian(
            nig_uniforms,
            nig_cache,
            nig_exact.inverse_gaussian_mean,
            nig_exact.inverse_gaussian_shape
        );
    const float nig_normal = philox::next_normal(nig_uniforms, nig_cache);
    normal_inverse_gaussian::one_step_transition(
        nig,
        nig_exact,
        nig_clock,
        nig_normal,
        nig_terminal_manual
    );

    const auto vg_initial_transition =
        variance_gamma::prepare_transition(vg, 0.1f);
    const auto vg_regular_transition =
        variance_gamma::prepare_transition(vg, 0.2f);
    const variance_gamma::PreparedTransition vg_calendar[3] = {
        vg_initial_transition,
        vg_regular_transition,
        vg_regular_transition,
    };
    float vg_regular_spots[3];
    float vg_calendar_spots[3];
    equity::SpotObservationWriter<variance_gamma::DynamicsPolicy>
        regular_writer{vg_regular_spots, 1U, 3U};
    equity::SpotObservationWriter<variance_gamma::DynamicsPolicy>
        calendar_writer{vg_calendar_spots, 1U, 3U};
    const auto vg_regular_state =
        equity::simulate_exact_transition_regular_schedule<
            variance_gamma::DynamicsPolicy
        >(
            vg,
            vg_initial_transition,
            vg_regular_transition,
            3U,
            key,
            31U,
            regular_writer
        );
    const auto vg_calendar_state =
        equity::simulate_exact_transition_calendar<
            variance_gamma::DynamicsPolicy
        >(vg, vg_calendar, 3U, key, 31U, calendar_writer);
    const std::uint32_t exact_regular_matches_calendar =
        static_cast<std::uint32_t>(
            vg_regular_state.log_spot == vg_calendar_state.log_spot
            && vg_regular_spots[0] == vg_calendar_spots[0]
            && vg_regular_spots[1] == vg_calendar_spots[1]
            && vg_regular_spots[2] == vg_calendar_spots[2]
        );

    *output = {
        vg,
        vg_transition_coefficients,
        vg_exact,
        nig,
        nig_transition_coefficients,
        nig_exact,
        vg_transition,
        nig_transition,
        gamma_small_shape_first,
        gamma_small_shape_replay,
        gamma_large_shape_first,
        gamma_large_shape_replay,
        inverse_gaussian_first,
        inverse_gaussian_replay,
        vg_terminal_first,
        vg_terminal_replay,
        vg_terminal_manual.log_spot,
        nig_terminal_first,
        nig_terminal_replay,
        nig_terminal_manual.log_spot,
        exact_regular_matches_calendar,
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(float lhs, float rhs, float tolerance = 3.0e-6f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
    using namespace ai_factory::workbench;

    constexpr float maturity = 0.5f;
    constexpr std::size_t step_count = 10U;
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "Levy dynamics test cudaGetDeviceCount");

    LevyDynamicsResults* device_results = nullptr;
    check_cuda(
        cudaMalloc(&device_results, sizeof(LevyDynamicsResults)),
        "Levy dynamics test cudaMalloc"
    );
    exercise_levy_dynamics_kernel<<<1, 1>>>(device_results);
    check_cuda(cudaGetLastError(), "Levy dynamics test kernel launch");
    LevyDynamicsResults results{};
    check_cuda(
        cudaMemcpy(
            &results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "Levy dynamics test cudaMemcpy"
    );
    check_cuda(cudaFree(device_results), "Levy dynamics test cudaFree");

    constexpr float dt = maturity / static_cast<float>(step_count);
    const float vg_argument =
        1.0f - (-0.1f) * 0.25f - 0.5f * 0.2f * 0.2f * 0.25f;
    const float vg_drift = (
        0.03f - 0.01f + std::log(vg_argument) / 0.25f
    ) * dt;
    require(close(results.vg_transition_coefficients.gamma_shape, dt / 0.25f)
                && results.vg.nu == 0.25f,
            "VG Gamma clock preparation is incorrect");
    require(close(results.vg_transition_coefficients.drift, vg_drift),
            "VG martingale correction is incorrect");
    require(close(results.vg_exact.gamma_shape, maturity / 0.25f)
                && close(results.vg_exact.drift, vg_drift * step_count),
            "VG exact-interval preparation is incorrect");
    const float expected_vg_transition = std::log(100.0f) + vg_drift
        + (-0.1f) * 0.08f + 0.2f * std::sqrt(0.08f) * -0.4f;
    require(close(results.vg_transition.log_spot, expected_vg_transition),
            "VG one-step transition is incorrect");

    const float gamma = std::sqrt(8.0f * 8.0f - 2.0f * 2.0f);
    const float delta_dt = 0.4f * dt;
    const float nig_correction = 0.4f * (
        std::sqrt(8.0f * 8.0f - 1.0f) - gamma
    );
    const float nig_drift = (0.03f - 0.01f + nig_correction) * dt;
    require(close(results.nig_transition_coefficients.inverse_gaussian_mean,
                  delta_dt / gamma)
                && close(results.nig_transition_coefficients.inverse_gaussian_shape,
                         delta_dt * delta_dt),
            "NIG inverse-Gaussian clock preparation is incorrect");
    require(close(results.nig_transition_coefficients.drift, nig_drift),
            "NIG martingale correction is incorrect");
    require(
        close(results.nig_exact.inverse_gaussian_mean,
              0.4f * maturity / gamma)
            && close(results.nig_exact.inverse_gaussian_shape,
                     0.4f * maturity * 0.4f * maturity)
            && close(results.nig_exact.drift, nig_drift * step_count),
        "NIG exact-interval preparation is incorrect"
    );
    const float expected_nig_transition = std::log(100.0f) + nig_drift
        + (-2.0f) * 0.03f + std::sqrt(0.03f) * 0.5f;
    require(close(results.nig_transition.log_spot, expected_nig_transition),
            "NIG one-step transition is incorrect");

    require(results.gamma_small_shape_first > 0.0f
                && results.gamma_large_shape_first > 0.0f
                && results.inverse_gaussian_first > 0.0f,
            "Levy distribution sampler returned a non-positive value");
    require(results.gamma_small_shape_first
                    == results.gamma_small_shape_replay
                && results.gamma_large_shape_first
                    == results.gamma_large_shape_replay
                && results.inverse_gaussian_first
                    == results.inverse_gaussian_replay,
            "Levy distribution samplers are not deterministic");
    require(results.vg_terminal_first == results.vg_terminal_replay
                && results.nig_terminal_first == results.nig_terminal_replay,
            "Levy terminal simulations are not deterministic");
    require(
        results.vg_terminal_first == results.vg_terminal_manual_aggregate
            && results.nig_terminal_first
                == results.nig_terminal_manual_aggregate,
        "Levy terminal simulations did not use exact interval increments"
    );
    require(
        results.exact_regular_matches_calendar == 1U,
        "Exact regular and calendar schedules do not match bit for bit"
    );
    return 0;
}
