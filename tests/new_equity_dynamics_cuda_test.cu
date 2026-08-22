// Verify the preparation, transition, and Philox replay contracts of the four
// equity dynamics added after the first catalogue release.
#include "common/check_cuda.cuh"
#include "common/equity/observation_handlers.cuh"
#include "common/equity/path_simulation.cuh"
#include "model/equity/cev/dynamics.cu"
#include "model/equity/kou/dynamics.cu"
#include "model/equity/merton/dynamics.cu"
#include "model/equity/schobel_zhu/dynamics.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

struct NewEquityDynamicsResults {
    ai_factory::workbench::merton::PreparedModel merton;
    ai_factory::workbench::merton::PreparedTransition merton_transition;
    ai_factory::workbench::kou::PreparedModel kou;
    ai_factory::workbench::kou::PreparedTransition kou_transition;
    ai_factory::workbench::cev::PreparedModel cev;
    ai_factory::workbench::schobel_zhu::PreparedModel
        schobel_zhu;
    float merton_terminal_first;
    float merton_terminal_replay;
    float kou_terminal_first;
    float kou_terminal_replay;
    float cev_terminal_first;
    float cev_terminal_replay;
    float schobel_zhu_terminal_first;
    float schobel_zhu_terminal_replay;
    float cev_transition;
    float cev_absorbed_transition;
    float schobel_zhu_volatility_transition;
    std::uint32_t cev_calendar_matches_regular;
    std::uint32_t schobel_zhu_calendar_matches_regular;
};

__global__ void exercise_new_equity_dynamics_kernel(
    NewEquityDynamicsResults* output
) {
    using namespace ai_factory::workbench;

    constexpr float maturity = 0.75f;
    constexpr std::uint32_t num_steps = 12U;
    constexpr float delta_t = maturity / static_cast<float>(num_steps);
    const merton::ModelParameters merton_parameters = {
        1.0f, 0.03f, 0.01f, 0.2f, 0.7f, -0.12f, 0.25f,
    };
    const kou::ModelParameters kou_parameters = {
        1.0f, 0.03f, 0.01f, 0.2f, 0.7f, 0.35f, 8.0f, 10.0f,
    };
    const cev::ModelParameters cev_parameters = {
        1.0f, 0.03f, 0.01f, 0.25f, 0.75f,
    };
    const schobel_zhu::ModelParameters schobel_zhu_parameters = {
        1.0f, 0.03f, 0.01f, 0.22f, 1.4f, 0.20f, 0.35f, -0.6f,
    };

    const auto merton_model = merton::prepare_model(merton_parameters);
    const auto merton_transition =
        merton::prepare_transition(merton_model, maturity);
    const auto kou_model = kou::prepare_model(kou_parameters);
    const auto kou_transition = kou::prepare_transition(kou_model, maturity);
    const auto cev_model = cev::prepare_model(cev_parameters, delta_t);
    const auto schobel_zhu_model = schobel_zhu::prepare_model(
        schobel_zhu_parameters,
        delta_t
    );
    const philox::PhiloxKey key = philox::make_key(900000001ULL);

    cev::State cev_transition = cev::initial_state(cev_model);
    cev::one_step_transition(cev_model, -0.35f, cev_transition);
    cev::State cev_absorbed_transition{0.0f};
    cev::one_step_transition(
        cev_model,
        0.75f,
        cev_absorbed_transition
    );

    schobel_zhu::State schobel_zhu_transition =
        schobel_zhu::initial_state(schobel_zhu_model);
    schobel_zhu::one_step_transition(
        schobel_zhu_model,
        -0.4f,
        0.3f,
        0.2f,
        schobel_zhu_transition
    );

    const float merton_terminal_first =
        equity::simulate_exact_transition_terminal<merton::DynamicsPolicy>(
            merton_model, merton_transition, key, 17U
        ).log_spot;
    const float merton_terminal_replay =
        equity::simulate_exact_transition_terminal<merton::DynamicsPolicy>(
            merton_model, merton_transition, key, 17U
        ).log_spot;
    const float kou_terminal_first =
        equity::simulate_exact_transition_terminal<kou::DynamicsPolicy>(
            kou_model, kou_transition, key, 19U
        ).log_spot;
    const float kou_terminal_replay =
        equity::simulate_exact_transition_terminal<kou::DynamicsPolicy>(
            kou_model, kou_transition, key, 19U
        ).log_spot;
    const float cev_terminal_first =
        equity::simulate_fixed_step_terminal<cev::DynamicsPolicy>(
            cev_model, num_steps, key, 23U
        ).spot;
    const float cev_terminal_replay =
        equity::simulate_fixed_step_terminal<cev::DynamicsPolicy>(
            cev_model, num_steps, key, 23U
        ).spot;
    const float schobel_zhu_terminal_first =
        equity::simulate_fixed_step_terminal<schobel_zhu::DynamicsPolicy>(
            schobel_zhu_model, num_steps, key, 29U
        ).log_spot;
    const float schobel_zhu_terminal_replay =
        equity::simulate_fixed_step_terminal<schobel_zhu::DynamicsPolicy>(
            schobel_zhu_model, num_steps, key, 29U
        ).log_spot;

    constexpr std::uint32_t steps_between_observations[3] = {2U, 5U, 5U};
    float cev_regular_observations[2];
    float cev_calendar_observations[2];
    equity::SpotObservationWriter<cev::DynamicsPolicy> cev_regular_recorder{
        cev_regular_observations,
        1U,
        2U,
    };
    const cev::State cev_regular =
        equity::simulate_fixed_step_regular_schedule<cev::DynamicsPolicy>(
            cev_model,
            2U,
            5U,
            3U,
            key,
            31U,
            cev_regular_recorder
        );
    equity::SpotObservationWriter<cev::DynamicsPolicy> cev_recorder{
        cev_calendar_observations,
        1U,
        2U,
    };
    const cev::State cev_calendar =
        equity::simulate_fixed_step_calendar<cev::DynamicsPolicy>(
            cev_model,
            steps_between_observations,
            3U,
            key,
            31U,
            cev_recorder
        );

    float schobel_zhu_regular_spots[2];
    float schobel_zhu_regular_volatilities[2];
    float schobel_zhu_calendar_spots[2];
    float schobel_zhu_calendar_volatilities[2];
    equity::SpotAndStateObservationWriter<
        schobel_zhu::DynamicsPolicy,
        &schobel_zhu::State::volatility
    >
        schobel_zhu_regular_recorder{
            schobel_zhu_regular_spots,
            schobel_zhu_regular_volatilities,
            1U,
            2U,
        };
    const schobel_zhu::State schobel_zhu_regular =
        equity::simulate_fixed_step_regular_schedule<
            schobel_zhu::DynamicsPolicy
        >(
            schobel_zhu_model,
            2U,
            5U,
            3U,
            key,
            37U,
            schobel_zhu_regular_recorder
        );
    equity::SpotAndStateObservationWriter<
        schobel_zhu::DynamicsPolicy,
        &schobel_zhu::State::volatility
    >
        schobel_zhu_recorder{
            schobel_zhu_calendar_spots,
            schobel_zhu_calendar_volatilities,
            1U,
            2U,
        };
    const schobel_zhu::State schobel_zhu_calendar =
        equity::simulate_fixed_step_calendar<schobel_zhu::DynamicsPolicy>(
            schobel_zhu_model,
            steps_between_observations,
            3U,
            key,
            37U,
            schobel_zhu_recorder
        );

    *output = {
        merton_model,
        merton_transition,
        kou_model,
        kou_transition,
        cev_model,
        schobel_zhu_model,
        merton_terminal_first,
        merton_terminal_replay,
        kou_terminal_first,
        kou_terminal_replay,
        cev_terminal_first,
        cev_terminal_replay,
        schobel_zhu_terminal_first,
        schobel_zhu_terminal_replay,
        cev_transition.spot,
        cev_absorbed_transition.spot,
        schobel_zhu_transition.volatility,
        static_cast<std::uint32_t>(
            cev_calendar.spot == cev_regular.spot
            && cev_calendar_observations[0] == cev_regular_observations[0]
            && cev_calendar_observations[1] == cev_regular_observations[1]
        ),
        static_cast<std::uint32_t>(
            schobel_zhu_calendar.log_spot == schobel_zhu_regular.log_spot
            && schobel_zhu_calendar.volatility
                == schobel_zhu_regular.volatility
            && schobel_zhu_calendar_spots[0]
                == schobel_zhu_regular_spots[0]
            && schobel_zhu_calendar_spots[1]
                == schobel_zhu_regular_spots[1]
            && schobel_zhu_calendar_volatilities[0]
                == schobel_zhu_regular_volatilities[0]
            && schobel_zhu_calendar_volatilities[1]
                == schobel_zhu_regular_volatilities[1]
        ),
    };
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(float left, float right, float tolerance = 3.0e-6f) {
    return std::fabs(left - right) <= tolerance;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    ai_factory::workbench::check_cuda(
        availability,
        "New equity dynamics test cudaGetDeviceCount"
    );

    NewEquityDynamicsResults* device_results = nullptr;
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device_results, sizeof(NewEquityDynamicsResults)),
        "New equity dynamics test cudaMalloc"
    );
    exercise_new_equity_dynamics_kernel<<<1, 1>>>(device_results);
    ai_factory::workbench::check_cuda(
        cudaGetLastError(),
        "New equity dynamics test kernel launch"
    );
    NewEquityDynamicsResults results{};
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            &results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "New equity dynamics test cudaMemcpy"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_results),
        "New equity dynamics test cudaFree"
    );

    constexpr float maturity = 0.75f;
    constexpr float dt = maturity / 12.0f;
    require(
        close(results.merton_transition.poisson_mean, 0.7f * maturity)
            && close(results.kou_transition.poisson_mean, 0.7f * maturity),
        "Jump-model Poisson interval preparation is incorrect"
    );
    require(
        close(results.cev.diffusion_scale, 0.25f * std::sqrt(dt))
            && close(results.cev.milstein_scale,
                     0.5f * 0.25f * 0.25f * 0.75f * dt),
        "CEV Milstein preparation is incorrect"
    );
    require(
        close(results.schobel_zhu.exp_mean_reversion_dt,
              std::exp(-1.4f * dt)),
        "Schobel-Zhu exact OU preparation is incorrect"
    );

    const float spot_beta = std::pow(1.0f, 0.75f);
    const float expected_cev_transition = std::max(
        0.0f,
        1.0f
            + (0.03f - 0.01f) * dt
            + 0.25f * std::sqrt(dt) * spot_beta * -0.35f
            + 0.5f * 0.25f * 0.25f * 0.75f * dt
                * (0.35f * 0.35f - 1.0f)
    );
    require(
        close(results.cev_transition, expected_cev_transition),
        "CEV one-step transition is incorrect"
    );
    require(
        results.cev_absorbed_transition == 0.0f,
        "CEV zero boundary is not absorbing"
    );

    const float expected_schobel_zhu_volatility =
        0.20f
        + (0.22f - 0.20f) * std::exp(-1.4f * dt)
        + results.schobel_zhu.ou_std * -0.4f;
    require(
        close(
            results.schobel_zhu_volatility_transition,
            expected_schobel_zhu_volatility
        ),
        "Schobel-Zhu exact OU endpoint transition is incorrect"
    );

    require(
        results.merton_terminal_first == results.merton_terminal_replay
            && results.kou_terminal_first == results.kou_terminal_replay
            && results.cev_terminal_first == results.cev_terminal_replay
            && results.schobel_zhu_terminal_first
                == results.schobel_zhu_terminal_replay,
        "New equity dynamics do not replay deterministically"
    );
    require(
        results.cev_calendar_matches_regular == 1U,
        "CEV calendar simulation differs from its equivalent regular grid"
    );
    require(
        results.schobel_zhu_calendar_matches_regular == 1U,
        "Schobel-Zhu calendar simulation differs from its regular grid"
    );
    return 0;
}
