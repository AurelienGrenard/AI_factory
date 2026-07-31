// Reusable Heston QE-M preparation and path simulation implementation.
#include "model/heston/dynamics.cuh"

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::heston {
namespace {

constexpr float kQePsiCritical = 1.5f;
constexpr float kGamma1 = 0.5f;
constexpr float kGamma2 = 0.5f;
constexpr std::uint64_t kUniformsPerStep = 3ULL;

}  // namespace

// Prepare coefficients that are constant across all paths of one result row.
__device__ __forceinline__ HestonQeParameters prepare_model(
    const HestonModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    const float kappa = parameters.kappa;
    const float theta = parameters.theta;
    const float gamma = parameters.gamma;
    const float rho = parameters.rho;
    const float dt = maturity / static_cast<float>(num_steps);
    const float exp_kdt = expf(-kappa * dt);
    const float one_minus_exp = 1.0f - exp_kdt;
    const float gamma2 = gamma * gamma;
    const float drift_dt =
        (parameters.risk_free_rate - parameters.dividend_yield) * dt;
    const float kappa_rho_over_gamma = kappa * rho / gamma;
    const float rho_over_gamma = rho / gamma;
    const float k2 =
        kGamma2 * dt * (kappa_rho_over_gamma - 0.5f) + rho_over_gamma;
    const float k4 = kGamma2 * dt * (1.0f - rho * rho);

    return {
        logf(parameters.spot),
        parameters.initial_variance,
        theta,
        exp_kdt,
        gamma2 * exp_kdt * one_minus_exp / kappa,
        theta * gamma2 * one_minus_exp * one_minus_exp / (2.0f * kappa),
        drift_dt,
        drift_dt - rho * kappa * theta * dt / gamma,
        kGamma1 * dt * (kappa_rho_over_gamma - 0.5f) - rho_over_gamma,
        k2,
        kGamma1 * dt * (1.0f - rho * rho),
        k4,
        k2 + 0.5f * k4,
    };
}

// Construct the time-zero state stored in the prepared QE parameters.
__device__ __forceinline__ HestonState initial_state(
    const HestonQeParameters& model
) {
    return {model.initial_log_spot, model.initial_variance};
}

// Apply one variance and log-spot update with the QE-M martingale correction.
__device__ __forceinline__ void one_step_transition(
    const HestonQeParameters& model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    HestonState& state
) {
    const float previous_variance = fmaxf(state.variance, 0.0f);
    const float conditional_mean =
        model.theta
        + (previous_variance - model.theta) * model.exp_kdt;
    const float conditional_variance =
        previous_variance * model.variance_linear_scale
        + model.variance_constant_scale;

    float next_variance = 0.0f;
    float log_moment = 0.0f;
    bool martingale_valid = true;

    if (conditional_mean > 0.0f && conditional_variance > 0.0f) {
        const float psi = conditional_variance
                        / (conditional_mean * conditional_mean);
        // Use the quadratic Gaussian branch for low conditional variance.
        if (psi <= kQePsiCritical) {
            const float inverse_psi = 1.0f / psi;
            const float root_term =
                fmaxf(2.0f * inverse_psi - 1.0f, 0.0f);
            const float b2 =
                root_term
                + sqrtf(2.0f * inverse_psi) * sqrtf(root_term);
            const float b = sqrtf(fmaxf(b2, 0.0f));
            const float a = conditional_mean / (1.0f + b2);
            const float shifted = b + variance_normal;
            next_variance = a * shifted * shifted;

            const float denominator =
                1.0f - 2.0f * model.martingale_a * a;
            if (denominator > 0.0f) {
                log_moment = model.martingale_a * b2 * a / denominator
                             - 0.5f * logf(denominator);
            } else {
                martingale_valid = false;
            }
        } else {
            // Use the mass-at-zero exponential branch for high variance.
            const float probability_zero = (psi - 1.0f) / (psi + 1.0f);
            const float beta =
                (1.0f - probability_zero) / conditional_mean;
            next_variance = variance_uniform <= probability_zero
                ? 0.0f
                : logf((1.0f - probability_zero) / (1.0f - variance_uniform))
                      / beta;

            if (model.martingale_a < beta) {
                const float moment =
                    probability_zero
                    + beta * (1.0f - probability_zero)
                          / (beta - model.martingale_a);
                martingale_valid = moment > 0.0f;
                if (martingale_valid) log_moment = logf(moment);
            } else {
                martingale_valid = false;
            }
        }
    }

    const float variance_integral_proxy = fmaxf(
        model.k3 * previous_variance + model.k4 * next_variance,
        0.0f
    );
    // Apply QE-M when valid, otherwise use the stable QE fallback.
    if (martingale_valid) {
        state.log_spot +=
            model.drift_dt - log_moment
            - 0.5f * model.k3 * previous_variance
            + model.k2 * next_variance
            + sqrtf(variance_integral_proxy) * stock_normal;
    } else {
        state.log_spot +=
            model.k0
            + model.k1 * previous_variance
            + model.k2 * next_variance
            + sqrtf(variance_integral_proxy) * stock_normal;
    }
    state.variance = next_variance;
}

// Generate all random variates for one path and return its terminal state.
__device__ __forceinline__ HestonState simulate_terminal_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    HestonState state = initial_state(model);
    const std::uint64_t uniform_count =
        kUniformsPerStep * static_cast<std::uint64_t>(num_steps);
    const std::uint64_t groups_per_path = (uniform_count + 3ULL) >> 2U;
    const std::uint64_t first_group =
        static_cast<std::uint64_t>(path) * groups_per_path;
    philox::UniformSequence uniforms(key, first_group);

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        const float angle_uniform = uniforms.next();
        const float radius_uniform = uniforms.next();
        const philox::NormalPair normals =
            philox::box_muller(angle_uniform, radius_uniform);
        const float variance_uniform = uniforms.next();

        one_step_transition(
            model,
            normals.first,
            variance_uniform,
            normals.second,
            state
        );
    }
    return state;
}

// Accumulate spots from time zero to maturity and return their arithmetic mean.
__device__ __forceinline__ HestonMeanPathResult simulate_mean_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    HestonState state = initial_state(model);
    const std::uint64_t uniform_count =
        kUniformsPerStep * static_cast<std::uint64_t>(num_steps);
    const std::uint64_t groups_per_path = (uniform_count + 3ULL) >> 2U;
    const std::uint64_t first_group =
        static_cast<std::uint64_t>(path) * groups_per_path;
    philox::UniformSequence uniforms(key, first_group);
    double spot_sum = static_cast<double>(expf(state.log_spot));

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        const float angle_uniform = uniforms.next();
        const float radius_uniform = uniforms.next();
        const philox::NormalPair normals =
            philox::box_muller(angle_uniform, radius_uniform);
        const float variance_uniform = uniforms.next();

        one_step_transition(
            model,
            normals.first,
            variance_uniform,
            normals.second,
            state
        );
        spot_sum += static_cast<double>(expf(state.log_spot));
    }

    return {
        state,
        static_cast<float>(
            spot_sum / (static_cast<double>(num_steps) + 1.0)
        ),
    };
}

// Track the maximum spot at time zero and after every simulated transition.
__device__ __forceinline__ HestonMaximumPathResult simulate_maximum_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    HestonState state = initial_state(model);
    const std::uint64_t uniform_count =
        kUniformsPerStep * static_cast<std::uint64_t>(num_steps);
    const std::uint64_t groups_per_path = (uniform_count + 3ULL) >> 2U;
    const std::uint64_t first_group =
        static_cast<std::uint64_t>(path) * groups_per_path;
    philox::UniformSequence uniforms(key, first_group);
    const float initial_spot = expf(state.log_spot);
    float maximum_spot = initial_spot;

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        const float angle_uniform = uniforms.next();
        const float radius_uniform = uniforms.next();
        const philox::NormalPair normals =
            philox::box_muller(angle_uniform, radius_uniform);
        const float variance_uniform = uniforms.next();

        one_step_transition(
            model,
            normals.first,
            variance_uniform,
            normals.second,
            state
        );
        const float spot = expf(state.log_spot);
        maximum_spot = fmaxf(maximum_spot, spot);
    }

    return {state, maximum_spot};
}

// Write pre-maturity states in a date-major grid and return terminal state.
__device__ __forceinline__ HestonState simulate_on_regular_grid(
    const HestonQeParameters& initial_stub_model,
    const HestonQeParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_variances
) {
    HestonState state = initial_state(initial_stub_model);
    const std::uint64_t total_steps =
        static_cast<std::uint64_t>(initial_stub_steps)
        + static_cast<std::uint64_t>(exercise_count - 1U)
            * static_cast<std::uint64_t>(steps_per_exercise);
    const std::uint64_t uniform_count = kUniformsPerStep * total_steps;
    const std::uint64_t groups_per_path = (uniform_count + 3ULL) >> 2U;
    const std::uint64_t first_group =
        static_cast<std::uint64_t>(path) * groups_per_path;
    philox::UniformSequence uniforms(key, first_group);

    // Reach the first exercise date through its possibly shorter stub.
    for (std::uint32_t step_index = 0U;
         step_index < initial_stub_steps;
         ++step_index) {
        const float angle_uniform = uniforms.next();
        const float radius_uniform = uniforms.next();
        const philox::NormalPair normals =
            philox::box_muller(angle_uniform, radius_uniform);
        const float variance_uniform = uniforms.next();

        one_step_transition(
            initial_stub_model,
            normals.first,
            variance_uniform,
            normals.second,
            state
        );
    }
    if (exercise_count > 1U) {
        observed_spots[path] = expf(state.log_spot);
        observed_variances[path] = state.variance;
    }

    // Every remaining exercise interval uses the same numerical grid.
    for (std::uint32_t exercise = 1U;
         exercise < exercise_count;
         ++exercise) {
        for (std::uint32_t step_index = 0U;
             step_index < steps_per_exercise;
             ++step_index) {
            const float angle_uniform = uniforms.next();
            const float radius_uniform = uniforms.next();
            const philox::NormalPair normals =
                philox::box_muller(angle_uniform, radius_uniform);
            const float variance_uniform = uniforms.next();

            one_step_transition(
                regular_model,
                normals.first,
                variance_uniform,
                normals.second,
                state
            );
        }

        if (exercise + 1U < exercise_count) {
            const std::size_t output_index =
                static_cast<std::size_t>(exercise) * path_count + path;
            observed_spots[output_index] = expf(state.log_spot);
            observed_variances[output_index] = state.variance;
        }
    }
    return state;
}

}  // namespace ai_factory::workbench::heston
