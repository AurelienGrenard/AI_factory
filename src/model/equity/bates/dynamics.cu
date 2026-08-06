// Reusable Bates QE-M preparation and path simulation implementation.
#include "model/equity/bates/dynamics.cuh"

#include "common/philox.cuh"

// Bates composes the existing QE-M variance/spot transition without changing
// Heston's implementation or its random-number mapping.
#include "model/equity/heston/dynamics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::bates {

// Prepare coefficients that are constant across all paths of one result row.
__device__ __forceinline__ BatesQeParameters prepare_model(
    const BatesModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    const float dt = maturity / static_cast<float>(num_steps);
    const heston::HestonModelParameters heston_parameters = {
        parameters.spot,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.initial_variance,
        parameters.kappa,
        parameters.theta,
        parameters.gamma,
        parameters.rho,
    };
    const float poisson_mean = parameters.jump_intensity * dt;
    const float expected_relative_jump = expm1f(
        fmaf(
            0.5f * parameters.jump_log_volatility,
            parameters.jump_log_volatility,
            parameters.jump_log_mean
        )
    );

    return {
        heston::prepare_model(heston_parameters, maturity, num_steps),
        poisson_mean,
        expf(-poisson_mean),
        parameters.jump_log_mean,
        parameters.jump_log_volatility,
        parameters.jump_intensity * expected_relative_jump * dt,
    };
}

// Construct the time-zero state stored in the prepared QE parameters.
__device__ __forceinline__ BatesState initial_state(
    const BatesQeParameters& model
) {
    return heston::initial_state(model.heston);
}

// Apply one variance and log-spot update with the QE-M martingale correction.
// Conditional on jump_count = n, the sum of n independent log jump sizes
// N(nu, delta^2) is represented exactly in law by
// n * nu + delta * sqrt(n) * jump_normal.  We therefore need neither the n
// individual sizes nor their event times to reproduce the process at grid
// dates.  This equivalence does not reconstruct the path inside the time step;
// continuously monitored or jump-time-dependent products need extra handling.
__device__ __forceinline__ void one_step_transition(
    const BatesQeParameters& model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    std::uint32_t jump_count,
    float jump_normal,
    BatesState& state
) {
    heston::one_step_transition(
        model.heston,
        variance_normal,
        variance_uniform,
        stock_normal,
        state
    );

    float jump_increment = -model.jump_compensator;
    if (jump_count != 0U) {
        const float count = static_cast<float>(jump_count);
        jump_increment = fmaf(count, model.jump_log_mean, jump_increment);
        jump_increment = fmaf(
            model.jump_log_volatility * sqrtf(count),
            jump_normal,
            jump_increment
        );
    }
    state.log_spot += jump_increment;
}

namespace {

// Draw one Bates transition from the continuous path-local uniform sequence.
// Conditional jump draws advance the same scalar stream without reservations.
__device__ __forceinline__ void simulate_one_step(
    const BatesQeParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    BatesState& state
) {
    const float variance_normal = philox::next_normal(
        uniforms, normal_cache
    );
    const float stock_normal = philox::next_normal(
        uniforms, normal_cache
    );
    const float variance_uniform = uniforms.next();
    const float poisson_uniform = uniforms.next();
    const std::uint32_t jump_count = philox::poisson_from_uniform(
        poisson_uniform,
        model.poisson_mean,
        model.poisson_zero_probability
    );

    float jump_normal = 0.0f;
    if (jump_count != 0U) {
        jump_normal = philox::next_normal(uniforms, normal_cache);
    }
    one_step_transition(
        model,
        variance_normal,
        variance_uniform,
        stock_normal,
        jump_count,
        jump_normal,
        state
    );
}

}  // namespace

// Generate all random variates for one path and return its terminal state.
__device__ __forceinline__ BatesState simulate_terminal_state(
    const BatesQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    BatesState state = initial_state(model);
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
__device__ __forceinline__ BatesMeanPathResult simulate_mean_state(
    const BatesQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    BatesState state = initial_state(model);
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
        state,
        static_cast<float>(
            spot_sum / (static_cast<double>(num_steps) + 1.0)
        ),
    };
}

// Average log-spots in FP64 and exponentiate only the completed mean.
__device__ __forceinline__ BatesGeometricMeanPathResult
simulate_geometric_mean_state(
    const BatesQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    BatesState state = initial_state(model);
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
        state,
        expf(static_cast<float>(log_spot_sum / observation_count)),
    };
}

// Reuse one Philox sequence across two exact QE-M interval preparations.
__device__ __forceinline__ BatesTwoTimePathResult simulate_at_two_times(
    const BatesQeParameters& first_model,
    const BatesQeParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t first_num_steps,
    std::size_t second_num_steps
) {
    BatesState state = initial_state(first_model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    for (std::size_t step_index = 0U;
         step_index < first_num_steps;
         ++step_index) {
        simulate_one_step(first_model, uniforms, normal_cache, state);
    }
    const BatesState first_state = state;

    for (std::size_t step_index = 0U;
         step_index < second_num_steps;
         ++step_index) {
        simulate_one_step(second_model, uniforms, normal_cache, state);
    }
    return {first_state, state};
}

// Track the maximum spot at time zero and after every simulated transition.
__device__ __forceinline__ BatesMaximumPathResult simulate_maximum_state(
    const BatesQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    BatesState state = initial_state(model);
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

    return {state, maximum_spot};
}

// Write pre-maturity states in a date-major grid and return terminal state.
__device__ __forceinline__ BatesState simulate_on_regular_grid(
    const BatesQeParameters& initial_stub_model,
    const BatesQeParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_variances
) {
    BatesState state = initial_state(initial_stub_model);
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

}  // namespace ai_factory::workbench::bates
