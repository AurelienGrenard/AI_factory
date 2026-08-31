// Verify preparation, transition, and Philox replay for four equity dynamics.
#include "common/check_cuda.cuh"
#include "common/equity/handlers.cuh"
#include "common/simulation/path_simulation.cuh"
#include "model/equity/markovian/cev/dynamics_impl.cuh"
#include "model/equity/markovian/kou/dynamics_impl.cuh"
#include "model/equity/markovian/merton/dynamics_impl.cuh"
#include "model/equity/markovian/schobel_zhu/dynamics_impl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

struct EquityDynamicsResults {
    ai_factory::workbench::model::equity::merton::PreparedModel merton;
    ai_factory::workbench::model::equity::merton::PreparedTransition merton_transition;
    ai_factory::workbench::model::equity::kou::PreparedModel kou;
    ai_factory::workbench::model::equity::kou::PreparedTransition kou_transition;
    ai_factory::workbench::model::equity::cev::PreparedModel cev;
    ai_factory::workbench::model::equity::schobel_zhu::PreparedModel
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

__global__ void exercise_equity_dynamics_kernel(
    EquityDynamicsResults* output
) {
    using namespace ai_factory::workbench;

    constexpr float maturity = 0.75f;
    constexpr std::uint32_t num_steps = 12U;
    constexpr float delta_t = maturity / static_cast<float>(num_steps);
    const ai_factory::workbench::model::equity::merton::ModelParameters merton_parameters = {
        1.0f, 0.03f, 0.01f, 0.2f, 0.7f, -0.12f, 0.25f,
    };
    const ai_factory::workbench::model::equity::kou::ModelParameters kou_parameters = {
        1.0f, 0.03f, 0.01f, 0.2f, 0.7f, 0.35f, 8.0f, 10.0f,
    };
    const ai_factory::workbench::model::equity::cev::ModelParameters cev_parameters = {
        1.0f, 0.03f, 0.01f, 0.25f, 0.75f,
    };
    const ai_factory::workbench::model::equity::schobel_zhu::ModelParameters schobel_zhu_parameters = {
        1.0f, 0.03f, 0.01f, 0.22f, 1.4f, 0.20f, 0.35f, -0.6f,
    };

    const auto merton_model = ai_factory::workbench::model::equity::merton::prepare_model(merton_parameters);
    const auto merton_transition =
        ai_factory::workbench::model::equity::merton::prepare_transition(merton_model, maturity);
    const auto kou_model = ai_factory::workbench::model::equity::kou::prepare_model(kou_parameters);
    const auto kou_transition = ai_factory::workbench::model::equity::kou::prepare_transition(kou_model, maturity);
    const auto cev_model = ai_factory::workbench::model::equity::cev::prepare_model(cev_parameters, delta_t);
    const auto schobel_zhu_model = ai_factory::workbench::model::equity::schobel_zhu::prepare_model(
        schobel_zhu_parameters,
        delta_t
    );
    const philox::PhiloxKey key = philox::make_key(900000001ULL);

    ai_factory::workbench::model::equity::cev::State cev_transition = ai_factory::workbench::model::equity::cev::initial_state(cev_model);
    ai_factory::workbench::model::equity::cev::one_step_transition(cev_model, -0.35f, cev_transition);
    ai_factory::workbench::model::equity::cev::State cev_absorbed_transition{0.0f};
    ai_factory::workbench::model::equity::cev::one_step_transition(
        cev_model,
        0.75f,
        cev_absorbed_transition
    );

    ai_factory::workbench::model::equity::schobel_zhu::State schobel_zhu_transition =
        ai_factory::workbench::model::equity::schobel_zhu::initial_state(schobel_zhu_model);
    ai_factory::workbench::model::equity::schobel_zhu::one_step_transition(
        schobel_zhu_model,
        -0.4f,
        0.3f,
        0.2f,
        schobel_zhu_transition
    );

    const float merton_terminal_first =
        simulation::simulate_exact_transition_terminal<ai_factory::workbench::model::equity::merton::DynamicsPolicy>(
            merton_model, merton_transition, key, 17U
        ).log_spot;
    const float merton_terminal_replay =
        simulation::simulate_exact_transition_terminal<ai_factory::workbench::model::equity::merton::DynamicsPolicy>(
            merton_model, merton_transition, key, 17U
        ).log_spot;
    const float kou_terminal_first =
        simulation::simulate_exact_transition_terminal<ai_factory::workbench::model::equity::kou::DynamicsPolicy>(
            kou_model, kou_transition, key, 19U
        ).log_spot;
    const float kou_terminal_replay =
        simulation::simulate_exact_transition_terminal<ai_factory::workbench::model::equity::kou::DynamicsPolicy>(
            kou_model, kou_transition, key, 19U
        ).log_spot;
    const float cev_terminal_first =
        simulation::simulate_fixed_step_terminal<ai_factory::workbench::model::equity::cev::DynamicsPolicy>(
            cev_model, num_steps, key, 23U
        ).spot;
    const float cev_terminal_replay =
        simulation::simulate_fixed_step_terminal<ai_factory::workbench::model::equity::cev::DynamicsPolicy>(
            cev_model, num_steps, key, 23U
        ).spot;
    const float schobel_zhu_terminal_first =
        simulation::simulate_fixed_step_terminal<ai_factory::workbench::model::equity::schobel_zhu::DynamicsPolicy>(
            schobel_zhu_model, num_steps, key, 29U
        ).log_spot;
    const float schobel_zhu_terminal_replay =
        simulation::simulate_fixed_step_terminal<ai_factory::workbench::model::equity::schobel_zhu::DynamicsPolicy>(
            schobel_zhu_model, num_steps, key, 29U
        ).log_spot;

    constexpr std::uint32_t steps_between_observations[3] = {2U, 5U, 5U};
    float cev_regular_observations[2];
    float cev_calendar_observations[2];
    equity::SpotObservationWriter<ai_factory::workbench::model::equity::cev::DynamicsPolicy> cev_regular_recorder{
        cev_regular_observations,
        1U,
        2U,
    };
    const ai_factory::workbench::model::equity::cev::State cev_regular =
        simulation::simulate_fixed_step_stubbed_regular_schedule<ai_factory::workbench::model::equity::cev::DynamicsPolicy>(
            cev_model,
            2U,
            5U,
            3U,
            key,
            31U,
            cev_regular_recorder
        );
    equity::SpotObservationWriter<ai_factory::workbench::model::equity::cev::DynamicsPolicy> cev_recorder{
        cev_calendar_observations,
        1U,
        2U,
    };
    const ai_factory::workbench::model::equity::cev::State cev_calendar =
        simulation::simulate_fixed_step_calendar<ai_factory::workbench::model::equity::cev::DynamicsPolicy>(
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
        ai_factory::workbench::model::equity::schobel_zhu::DynamicsPolicy,
        &ai_factory::workbench::model::equity::schobel_zhu::State::volatility
    >
        schobel_zhu_regular_recorder{
            schobel_zhu_regular_spots,
            schobel_zhu_regular_volatilities,
            1U,
            2U,
        };
    const ai_factory::workbench::model::equity::schobel_zhu::State schobel_zhu_regular =
        simulation::simulate_fixed_step_stubbed_regular_schedule<
            ai_factory::workbench::model::equity::schobel_zhu::DynamicsPolicy
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
        ai_factory::workbench::model::equity::schobel_zhu::DynamicsPolicy,
        &ai_factory::workbench::model::equity::schobel_zhu::State::volatility
    >
        schobel_zhu_recorder{
            schobel_zhu_calendar_spots,
            schobel_zhu_calendar_volatilities,
            1U,
            2U,
        };
    const ai_factory::workbench::model::equity::schobel_zhu::State schobel_zhu_calendar =
        simulation::simulate_fixed_step_calendar<ai_factory::workbench::model::equity::schobel_zhu::DynamicsPolicy>(
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
        "Merton/Kou/CEV/Schobel-Zhu dynamics contract cudaGetDeviceCount"
    );

    EquityDynamicsResults* device_results = nullptr;
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device_results, sizeof(EquityDynamicsResults)),
        "Merton/Kou/CEV/Schobel-Zhu dynamics contract cudaMalloc"
    );
    exercise_equity_dynamics_kernel<<<1, 1>>>(device_results);
    ai_factory::workbench::check_cuda(
        cudaGetLastError(),
        "Merton/Kou/CEV/Schobel-Zhu dynamics contract kernel launch"
    );
    EquityDynamicsResults results{};
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            &results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "Merton/Kou/CEV/Schobel-Zhu dynamics contract cudaMemcpy"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_results),
        "Merton/Kou/CEV/Schobel-Zhu dynamics contract cudaFree"
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
        close(results.schobel_zhu.volatility_decay,
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
        + results.schobel_zhu.volatility_standard_deviation * -0.4f;
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
        "Equity dynamics do not replay deterministically"
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
