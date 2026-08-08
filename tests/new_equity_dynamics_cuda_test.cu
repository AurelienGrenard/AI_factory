// Verify the preparation, transition, and Philox replay contracts of the four
// equity dynamics added after the first catalogue release.
#include "common/check_cuda.cuh"
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
    ai_factory::workbench::merton::MertonPreparedParameters merton;
    ai_factory::workbench::kou::KouPreparedParameters kou;
    ai_factory::workbench::cev::CevPreparedParameters cev;
    ai_factory::workbench::schobel_zhu::SchobelZhuPreparedParameters
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
};

__global__ void exercise_new_equity_dynamics_kernel(
    NewEquityDynamicsResults* output
) {
    using namespace ai_factory::workbench;

    constexpr float maturity = 0.75f;
    constexpr std::size_t num_steps = 12U;
    const merton::MertonModelParameters merton_parameters = {
        1.0f, 0.03f, 0.01f, 0.2f, 0.7f, -0.12f, 0.25f,
    };
    const kou::KouModelParameters kou_parameters = {
        1.0f, 0.03f, 0.01f, 0.2f, 0.7f, 0.35f, 8.0f, 10.0f,
    };
    const cev::CevModelParameters cev_parameters = {
        1.0f, 0.03f, 0.01f, 0.25f, 0.75f,
    };
    const schobel_zhu::SchobelZhuModelParameters schobel_zhu_parameters = {
        1.0f, 0.03f, 0.01f, 0.22f, 1.4f, 0.20f, 0.35f, -0.6f,
    };

    const auto merton_model = merton::prepare_model(
        merton_parameters,
        maturity
    );
    const auto kou_model = kou::prepare_model(kou_parameters, maturity);
    const auto cev_model = cev::prepare_model(
        cev_parameters,
        maturity,
        num_steps
    );
    const auto schobel_zhu_model = schobel_zhu::prepare_model(
        schobel_zhu_parameters,
        maturity,
        num_steps
    );
    const philox::PhiloxKey key = philox::make_key(900000001ULL);

    cev::CevState cev_transition = cev::initial_state(cev_model);
    cev::one_step_transition(cev_model, -0.35f, cev_transition);
    cev::CevState cev_absorbed_transition{0.0f};
    cev::one_step_transition(
        cev_model,
        0.75f,
        cev_absorbed_transition
    );

    schobel_zhu::SchobelZhuState schobel_zhu_transition =
        schobel_zhu::initial_state(schobel_zhu_model);
    schobel_zhu::one_step_transition(
        schobel_zhu_model,
        -0.4f,
        0.3f,
        0.2f,
        schobel_zhu_transition
    );

    const float merton_terminal_first = merton::simulate_terminal_state(
        merton_model,
        key,
        17U
    ).log_spot;
    const float merton_terminal_replay = merton::simulate_terminal_state(
        merton_model,
        key,
        17U
    ).log_spot;
    const float kou_terminal_first = kou::simulate_terminal_state(
        kou_model,
        key,
        19U
    ).log_spot;
    const float kou_terminal_replay = kou::simulate_terminal_state(
        kou_model,
        key,
        19U
    ).log_spot;
    const float cev_terminal_first = cev::simulate_terminal_state(
        cev_model,
        key,
        23U,
        num_steps
    ).spot;
    const float cev_terminal_replay = cev::simulate_terminal_state(
        cev_model,
        key,
        23U,
        num_steps
    ).spot;
    const float schobel_zhu_terminal_first =
        schobel_zhu::simulate_terminal_state(
            schobel_zhu_model,
            key,
            29U,
            num_steps
        ).log_spot;
    const float schobel_zhu_terminal_replay =
        schobel_zhu::simulate_terminal_state(
            schobel_zhu_model,
            key,
            29U,
            num_steps
        ).log_spot;

    *output = {
        merton_model,
        kou_model,
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
        close(results.merton.poisson_mean, 0.7f * maturity)
            && close(results.kou.poisson_mean, 0.7f * maturity),
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
    return 0;
}
