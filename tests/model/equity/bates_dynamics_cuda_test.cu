// Verify Bates preparation and the explicit one-step jump composition.
#include "common/check_cuda.cuh"
#include "common/equity/handlers.cuh"
#include "common/simulation/path_simulation.cuh"
#include "model/equity/markovian/bates/dynamics_impl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

namespace bates = ai_factory::workbench::model::equity::bates;
namespace heston = ai_factory::workbench::model::equity::heston;

struct DynamicsResults {
    ai_factory::workbench::model::equity::bates::PreparedModel prepared;
    ai_factory::workbench::model::equity::bates::State initial;
    ai_factory::workbench::model::equity::bates::State no_jump;
    ai_factory::workbench::model::equity::heston::State heston_no_jump;
    ai_factory::workbench::model::equity::bates::State with_jumps;
    ai_factory::workbench::model::equity::heston::State heston_before_jumps;
    ai_factory::workbench::model::equity::bates::State terminal_first;
    ai_factory::workbench::model::equity::bates::State terminal_second;
    ai_factory::workbench::model::equity::bates::State no_jump_terminal;
    ai_factory::workbench::model::equity::heston::State heston_terminal;
    std::uint32_t poisson_zero;
    std::uint32_t poisson_one;
    std::uint32_t poisson_two;
    std::uint32_t heston_calendar_matches_regular;
    std::uint32_t bates_calendar_matches_regular;
};

__global__ void exercise_bates_dynamics_kernel(DynamicsResults* output) {
    using namespace ai_factory::workbench;

    constexpr float maturity = 0.5f;
    constexpr std::uint32_t step_count = 10U;
    constexpr float delta_t = maturity / static_cast<float>(step_count);
    const ai_factory::workbench::model::equity::bates::ModelParameters parameters = {
        1.25f, 0.03f, 0.01f, 0.05f, 1.4f, 0.04f, 0.32f, -0.65f,
        0.8f, -0.12f, 0.24f,
    };
    const ai_factory::workbench::model::equity::bates::ModelParameters no_jump_parameters = {
        1.25f, 0.03f, 0.01f, 0.05f, 1.4f, 0.04f, 0.32f, -0.65f,
        0.0f, -0.12f, 0.24f,
    };
    const ai_factory::workbench::model::equity::bates::PreparedModel prepared =
        ai_factory::workbench::model::equity::bates::prepare_model(parameters, delta_t);
    const ai_factory::workbench::model::equity::bates::PreparedModel no_jump_prepared =
        ai_factory::workbench::model::equity::bates::prepare_model(no_jump_parameters, delta_t);

    ai_factory::workbench::model::equity::bates::State no_jump = ai_factory::workbench::model::equity::bates::initial_state(no_jump_prepared);
    ai_factory::workbench::model::equity::heston::State heston_no_jump = no_jump;
    constexpr float variance_normal = -0.35f;
    constexpr float variance_uniform = 0.62f;
    constexpr float stock_normal = 0.41f;
    ai_factory::workbench::model::equity::bates::one_step_transition(
        no_jump_prepared,
        variance_normal,
        variance_uniform,
        stock_normal,
        0U,
        0.0f,
        no_jump
    );
    ai_factory::workbench::model::equity::heston::one_step_transition(
        no_jump_prepared.heston,
        variance_normal,
        variance_uniform,
        stock_normal,
        heston_no_jump
    );

    ai_factory::workbench::model::equity::bates::State with_jumps = ai_factory::workbench::model::equity::bates::initial_state(prepared);
    ai_factory::workbench::model::equity::heston::State heston_before_jumps = with_jumps;
    ai_factory::workbench::model::equity::heston::one_step_transition(
        prepared.heston,
        variance_normal,
        variance_uniform,
        stock_normal,
        heston_before_jumps
    );
    ai_factory::workbench::model::equity::bates::one_step_transition(
        prepared,
        variance_normal,
        variance_uniform,
        stock_normal,
        3U,
        -0.27f,
        with_jumps
    );

    const philox::PhiloxKey key = philox::make_key(900000001ULL);
    const ai_factory::workbench::model::equity::bates::State terminal_first =
        simulation::simulate_fixed_step_terminal<ai_factory::workbench::model::equity::bates::DynamicsPolicy>(
            prepared, step_count, key, 17U
        );
    const ai_factory::workbench::model::equity::bates::State terminal_second =
        simulation::simulate_fixed_step_terminal<ai_factory::workbench::model::equity::bates::DynamicsPolicy>(
            prepared, step_count, key, 17U
        );
    const ai_factory::workbench::model::equity::bates::State no_jump_terminal =
        simulation::simulate_fixed_step_terminal<ai_factory::workbench::model::equity::bates::DynamicsPolicy>(
            no_jump_prepared, step_count, key, 19U
        );
    const ai_factory::workbench::model::equity::heston::State heston_terminal =
        simulation::simulate_fixed_step_terminal<ai_factory::workbench::model::equity::heston::DynamicsPolicy>(
            no_jump_prepared.heston, step_count, key, 19U
        );
    const std::uint32_t poisson_zero = philox::poisson_from_uniform(
        0.5f, prepared.poisson_mean, prepared.zero_jump_probability
    );
    const std::uint32_t poisson_one = philox::poisson_from_uniform(
        0.98f, prepared.poisson_mean, prepared.zero_jump_probability
    );
    const std::uint32_t poisson_two = philox::poisson_from_uniform(
        0.9999f, prepared.poisson_mean, prepared.zero_jump_probability
    );

    constexpr std::uint32_t steps_between_observations[3] = {2U, 4U, 4U};
    float heston_regular_spots[2];
    float heston_regular_variances[2];
    float heston_calendar_spots[2];
    float heston_calendar_variances[2];
    equity::SpotAndStateObservationWriter<
        ai_factory::workbench::model::equity::heston::DynamicsPolicy,
        &ai_factory::workbench::model::equity::heston::State::variance
    >
        heston_regular_recorder{
            heston_regular_spots,
            heston_regular_variances,
            1U,
            2U,
        };
    const ai_factory::workbench::model::equity::heston::State heston_regular =
        simulation::simulate_fixed_step_stubbed_regular_schedule<ai_factory::workbench::model::equity::heston::DynamicsPolicy>(
            no_jump_prepared.heston,
            2U,
            4U,
            3U,
            key,
            23U,
            heston_regular_recorder
        );
    equity::SpotAndStateObservationWriter<
        ai_factory::workbench::model::equity::heston::DynamicsPolicy,
        &ai_factory::workbench::model::equity::heston::State::variance
    > heston_recorder{
        heston_calendar_spots,
        heston_calendar_variances,
        1U,
        2U,
    };
    const ai_factory::workbench::model::equity::heston::State heston_calendar =
        simulation::simulate_fixed_step_calendar<ai_factory::workbench::model::equity::heston::DynamicsPolicy>(
            no_jump_prepared.heston,
            steps_between_observations,
            3U,
            key,
            23U,
            heston_recorder
        );

    float bates_regular_spots[2];
    float bates_regular_variances[2];
    float bates_calendar_spots[2];
    float bates_calendar_variances[2];
    equity::SpotAndStateObservationWriter<
        ai_factory::workbench::model::equity::bates::DynamicsPolicy,
        &ai_factory::workbench::model::equity::bates::State::variance
    >
        bates_regular_recorder{
            bates_regular_spots,
            bates_regular_variances,
            1U,
            2U,
        };
    const ai_factory::workbench::model::equity::bates::State bates_regular =
        simulation::simulate_fixed_step_stubbed_regular_schedule<ai_factory::workbench::model::equity::bates::DynamicsPolicy>(
            prepared,
            2U,
            4U,
            3U,
            key,
            29U,
            bates_regular_recorder
        );
    equity::SpotAndStateObservationWriter<
        ai_factory::workbench::model::equity::bates::DynamicsPolicy,
        &ai_factory::workbench::model::equity::bates::State::variance
    > bates_recorder{
        bates_calendar_spots,
        bates_calendar_variances,
        1U,
        2U,
    };
    const ai_factory::workbench::model::equity::bates::State bates_calendar =
        simulation::simulate_fixed_step_calendar<ai_factory::workbench::model::equity::bates::DynamicsPolicy>(
            prepared,
            steps_between_observations,
            3U,
            key,
            29U,
            bates_recorder
        );
    *output = {
        prepared,
        ai_factory::workbench::model::equity::bates::initial_state(prepared),
        no_jump,
        heston_no_jump,
        with_jumps,
        heston_before_jumps,
        terminal_first,
        terminal_second,
        no_jump_terminal,
        heston_terminal,
        poisson_zero,
        poisson_one,
        poisson_two,
        static_cast<std::uint32_t>(
            heston_calendar.log_spot == heston_regular.log_spot
            && heston_calendar.variance == heston_regular.variance
            && heston_calendar_spots[0] == heston_regular_spots[0]
            && heston_calendar_spots[1] == heston_regular_spots[1]
            && heston_calendar_variances[0] == heston_regular_variances[0]
            && heston_calendar_variances[1] == heston_regular_variances[1]
        ),
        static_cast<std::uint32_t>(
            bates_calendar.log_spot == bates_regular.log_spot
            && bates_calendar.variance == bates_regular.variance
            && bates_calendar_spots[0] == bates_regular_spots[0]
            && bates_calendar_spots[1] == bates_regular_spots[1]
            && bates_calendar_variances[0] == bates_regular_variances[0]
            && bates_calendar_variances[1] == bates_regular_variances[1]
        ),
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(float lhs, float rhs, float tolerance = 2.0e-6f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

__global__ void jump_interval_probe(float mean, std::uint32_t steps, float* counts) {
    using namespace ai_factory::workbench;
    const unsigned int path = blockIdx.x * blockDim.x + threadIdx.x;
    bates::ModelParameters parameters{
        1.0f, 0.0f, 0.0f, 0.04f, 1.0f, 0.04f, 0.2f, -0.5f,
        mean * 252.0f / static_cast<float>(steps), -0.01f, 0.0f,
    };
    const auto prepared = bates::prepare_model(parameters, 1.0f / 252.0f);
    auto state = bates::initial_state(prepared);
    philox::UniformSequence uniforms(philox::make_key(819273ULL), path);
    philox::NormalPairCache cache;
    bates::simulate_jump_interval(prepared, steps, uniforms, cache, state);
    counts[path] = (state.log_spot + prepared.jump_compensator * steps)
        / parameters.jump_log_mean;
}

void check_aggregated_jump_law() {
    using ai_factory::workbench::check_cuda;
    constexpr unsigned int paths = 32768U;
    std::vector<float> counts(paths);
    float* device = nullptr;
    check_cuda(cudaMalloc(&device, paths * sizeof(float)), "Bates jump allocation");
    for (const std::uint32_t steps : {1U, 252U}) {
        for (const float mean : {0.0f, 0.1f, 9.99f, 10.0f, 110.0f, 1000.0f}) {
            jump_interval_probe<<<paths / 128U, 128U>>>(mean, steps, device);
            check_cuda(cudaGetLastError(), "Bates jump launch");
            check_cuda(cudaMemcpy(counts.data(), device, paths * sizeof(float),
                                  cudaMemcpyDeviceToHost), "Bates jump copy");
            double sum = 0.0, squares = 0.0;
            for (const float count : counts) {
                require(std::isfinite(count) && count >= -1.0e-3f
                            && std::abs(count - std::round(count)) < 1.0e-3f,
                        "Bates jump count is not a nonnegative integer");
                sum += count;
                squares += static_cast<double>(count) * count;
            }
            const double average = sum / paths;
            const double variance = squares / paths - average * average;
            // Poisson moments, with six-standard-error statistical budgets.
            require(std::abs(average - mean) <= 6.0 * std::sqrt(mean / paths) + 1.0e-4,
                    "Bates aggregated jump mean is incorrect");
            require(std::abs(variance - mean)
                        <= 6.0 * std::sqrt((mean + 2.0 * mean * mean) / paths) + 1.0e-3,
                    "Bates aggregated jump variance is incorrect");
            // Independent host Poisson CDF and compensated multiplicative moment.
            for (const double z : {-2.0, 0.0, 2.0}) {
                const int cutoff = static_cast<int>(std::floor(mean + z * std::sqrt(mean)));
                double expected = 0.0, observed = 0.0;
                for (int k = 0; k <= cutoff; ++k) {
                    expected += mean == 0.0f ? (k == 0 ? 1.0 : 0.0)
                        : std::exp(-mean + k * std::log(static_cast<double>(mean)) - std::lgamma(k + 1.0));
                }
                for (const float count : counts) observed += std::round(count) <= cutoff;
                expected = std::clamp(expected, 0.0, 1.0);
                require(std::abs(observed / paths - expected)
                            <= 6.0 * std::sqrt(expected * (1.0 - expected) / paths) + 2.0 / paths,
                        "Bates aggregated jump CDF is incorrect");
            }
            double compensated_mean = 0.0;
            for (const float count : counts) {
                compensated_mean += std::exp(-0.01 * std::round(count) - mean * std::expm1(-0.01));
            }
            const double compensated_variance = std::expm1(
                mean * (std::expm1(-0.02) - 2.0 * std::expm1(-0.01)));
            require(std::abs(compensated_mean / paths - 1.0)
                        <= 6.0 * std::sqrt(compensated_variance / paths) + 1.0e-5,
                    "Bates aggregated jump compensation is incorrect");
            std::vector<float> replay(paths);
            jump_interval_probe<<<paths / 256U, 256U>>>(mean, steps, device);
            check_cuda(cudaGetLastError(), "Bates jump replay launch");
            check_cuda(cudaMemcpy(replay.data(), device, paths * sizeof(float),
                                  cudaMemcpyDeviceToHost), "Bates jump replay copy");
            require(replay == counts, "Bates jump replay depends on launch geometry");
        }
    }
    check_cuda(cudaFree(device), "Bates jump release");
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
    check_aggregated_jump_law();

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
    require(close(results.prepared.zero_jump_probability,
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
    require(
        results.no_jump_terminal.log_spot == results.heston_terminal.log_spot
            && results.no_jump_terminal.variance
                == results.heston_terminal.variance,
        "Aggregated zero-intensity Bates terminal path differs from Heston"
    );
    require(results.poisson_zero == 0U
                && results.poisson_one == 1U
                && results.poisson_two == 2U,
            "Philox Poisson inverse CDF returned incorrect quantiles");
    require(
        results.heston_calendar_matches_regular == 1U,
        "Heston calendar simulation differs from its equivalent regular grid"
    );
    require(
        results.bates_calendar_matches_regular == 1U,
        "Bates calendar simulation differs from its equivalent regular grid"
    );
    return 0;
}
