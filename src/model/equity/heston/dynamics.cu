// Reusable Heston QE-M preparation and path simulation implementation.
#include "model/equity/heston/dynamics.cuh"

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::heston {

// ==================== Model-specific implementation =======================

namespace {

constexpr float kQePsiCritical = 1.5f;
constexpr float kGamma1 = 0.5f;
constexpr float kGamma2 = 0.5f;

}  // namespace

// ======================== Common equity dynamics =========================

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
    const float one_minus_exp = -expm1f(-kappa * dt);
    const float exp_kdt = 1.0f - one_minus_exp;
    const float gamma2 = gamma * gamma;
    const float drift_dt = fmaf(
        parameters.risk_free_rate - parameters.dividend_yield,
        dt,
        0.0f
    );
    const float kappa_rho_over_gamma = kappa * rho / gamma;
    const float rho_over_gamma = rho / gamma;
    const float k2 = fmaf(
        kGamma2 * dt,
        kappa_rho_over_gamma - 0.5f,
        rho_over_gamma
    );
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
    const float conditional_mean = fmaf(
        previous_variance - model.theta,
        model.exp_kdt,
        model.theta
    );
    const float conditional_variance = fmaf(
        previous_variance,
        model.variance_linear_scale,
        model.variance_constant_scale
    );

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
                root_term + sqrtf(2.0f * inverse_psi * root_term);
            const float b = sqrtf(b2);
            const float a = conditional_mean / (1.0f + b2);
            const float shifted = b + variance_normal;
            next_variance = a * shifted * shifted;

            const float denominator = fmaf(
                -2.0f * model.martingale_a, a, 1.0f
            );
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
        fmaf(model.k3, previous_variance, model.k4 * next_variance),
        0.0f
    );
    const float stock_diffusion =
        sqrtf(variance_integral_proxy) * stock_normal;
    // Apply QE-M when valid, otherwise use the stable QE fallback.
    if (martingale_valid) {
        float increment = fmaf(
            -0.5f * model.k3,
            previous_variance,
            model.drift_dt - log_moment
        );
        increment = fmaf(model.k2, next_variance, increment);
        state.log_spot += increment + stock_diffusion;
    } else {
        float increment = fmaf(
            model.k1, previous_variance, model.k0
        );
        increment = fmaf(model.k2, next_variance, increment);
        state.log_spot += increment + stock_diffusion;
    }
    state.variance = next_variance;
}

// ==================== Model-specific implementation =======================

namespace {

// Draw the three variates consumed by one fused QE-M transition.
__device__ __forceinline__ void simulate_one_step(
    const HestonQeParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    HestonState& state
) {
    const float variance_normal = philox::next_normal(
        uniforms, normal_cache
    );
    const float stock_normal = philox::next_normal(
        uniforms, normal_cache
    );
    const float variance_uniform = uniforms.next();
    one_step_transition(
        model,
        variance_normal,
        variance_uniform,
        stock_normal,
        state
    );
}

}  // namespace

// ======================== Common equity dynamics =========================

// Generate all random variates for one path and return its terminal state.
__device__ __forceinline__ HestonState simulate_terminal_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    HestonState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
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
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    double spot_sum = static_cast<double>(expf(state.log_spot));

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
        spot_sum += static_cast<double>(expf(state.log_spot));
    }

    return {
        static_cast<float>(
            spot_sum / (static_cast<double>(num_steps) + 1.0)
        ),
    };
}

// Average log-spots in FP64 and exponentiate only the completed mean.
__device__ __forceinline__ HestonGeometricMeanPathResult
simulate_geometric_mean_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    HestonState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    double log_spot_sum = static_cast<double>(state.log_spot);

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
        log_spot_sum += static_cast<double>(state.log_spot);
    }

    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {
        expf(static_cast<float>(log_spot_sum / observation_count)),
    };
}

// Reuse one Philox sequence across two exact QE-M interval preparations.
__device__ __forceinline__ HestonTwoTimePathResult simulate_at_two_times(
    const HestonQeParameters& first_model,
    const HestonQeParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t first_num_steps,
    std::size_t second_num_steps
) {
    HestonState state = initial_state(first_model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    for (std::size_t step_index = 0U;
         step_index < first_num_steps;
         ++step_index) {
        simulate_one_step(first_model, uniforms, normal_cache, state);
    }
    const float first_spot = expf(state.log_spot);

    for (std::size_t step_index = 0U;
         step_index < second_num_steps;
         ++step_index) {
        simulate_one_step(second_model, uniforms, normal_cache, state);
    }
    return {first_spot, expf(state.log_spot)};
}

// Track the maximum spot at time zero and after every simulated transition.
__device__ __forceinline__ HestonMaximumPathResult simulate_maximum_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    HestonState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    const float initial_spot = expf(state.log_spot);
    float maximum_spot = initial_spot;

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
        const float spot = expf(state.log_spot);
        maximum_spot = fmaxf(maximum_spot, spot);
    }

    return {maximum_spot};
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
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    // Reach the first exercise date through its possibly shorter stub.
    for (std::uint32_t step_index = 0U;
         step_index < initial_stub_steps;
         ++step_index) {
        simulate_one_step(initial_stub_model, uniforms, normal_cache, state);
    }
    if (exercise_count == 1U) return state;
    std::size_t output_index = path;
    observed_spots[output_index] = expf(state.log_spot);
    observed_variances[output_index] = state.variance;

    // Store only pre-terminal states with one running date-major offset.
    for (std::uint32_t exercise = 1U;
         exercise + 1U < exercise_count;
         ++exercise) {
        for (std::uint32_t step_index = 0U;
             step_index < steps_per_exercise;
             ++step_index) {
            simulate_one_step(regular_model, uniforms, normal_cache, state);
        }
        output_index += path_count;
        observed_spots[output_index] = expf(state.log_spot);
        observed_variances[output_index] = state.variance;
    }

    // Simulate the maturity interval without a global-memory write.
    for (std::uint32_t step_index = 0U;
         step_index < steps_per_exercise;
         ++step_index) {
        simulate_one_step(regular_model, uniforms, normal_cache, state);
    }
    return state;
}

}  // namespace ai_factory::workbench::heston
