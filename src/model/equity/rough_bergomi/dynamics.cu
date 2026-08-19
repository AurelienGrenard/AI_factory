// Flat-xi0 rough-Bergomi simulation with the kappa=1 hybrid scheme.
#include "model/equity/rough_bergomi/dynamics.cuh"

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::rough_bergomi {

__device__ __forceinline__ RoughBergomiPreparedParameters prepare_model(
    const RoughBergomiModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    const float step_count = static_cast<float>(num_steps);
    const float time_step = maturity / step_count;
    const float h = parameters.hurst_exponent;
    const float alpha = h - 0.5f;
    const float alpha_plus_one = h + 0.5f;
    const float two_h = 2.0f * h;
    const float time_step_to_h = powf(time_step, h);

    // For I = integral_0^dt (dt-s)^alpha dW_s and dW = sqrt(dt) Z:
    // Cov(I,Z) = dt^H / (H+1/2), Var(I) = dt^(2H) / (2H).
    const float singular_driver_loading =
        time_step_to_h / alpha_plus_one;
    const float singular_variance =
        time_step_to_h * time_step_to_h / two_h;
    const float independent_variance = fmaxf(
        singular_variance
            - singular_driver_loading * singular_driver_loading,
        0.0f
    );

    return {
        logf(parameters.spot),
        parameters.xi_0,
        logf(parameters.xi_0),
        parameters.eta,
        0.5f * parameters.eta * parameters.eta,
        alpha_plus_one,
        two_h,
        sqrtf(two_h),
        time_step,
        sqrtf(time_step),
        powf(time_step, alpha),
        (parameters.risk_free_rate - parameters.dividend_yield) * time_step,
        parameters.rho,
        sqrtf(fmaxf(1.0f - parameters.rho * parameters.rho, 0.0f)),
        singular_driver_loading,
        sqrtf(independent_variance),
    };
}

__device__ __forceinline__ void prepare_hybrid_grid(
    const RoughBergomiPreparedParameters& model,
    std::uint32_t step_count,
    float* far_weights,
    float* log_variance_corrections,
    std::uint32_t thread_index,
    std::uint32_t thread_count
) {
    for (std::uint32_t index = thread_index;
         index < step_count;
         index += thread_count) {
        // far_weights[index] represents lag k=index+2. The last slot is
        // intentionally unused so both shared arrays keep step_count entries.
        const std::uint32_t lag = index + 2U;
        if (lag <= step_count) {
            const float lag_value = static_cast<float>(lag);
            const float previous_lag = static_cast<float>(lag - 1U);
            const float optimal_cell_average = (
                powf(lag_value, model.alpha_plus_one)
                - powf(previous_lag, model.alpha_plus_one)
            ) / model.alpha_plus_one;
            far_weights[index] =
                model.time_step_to_alpha * optimal_cell_average;
        } else {
            far_weights[index] = 0.0f;
        }

        const float time =
            static_cast<float>(index + 1U) * model.time_step;
        log_variance_corrections[index] =
            model.log_initial_variance
            - model.half_eta_squared * powf(time, model.two_h);
    }
}

__device__ __forceinline__ RoughBergomiState initial_state(
    const RoughBergomiPreparedParameters& model
) {
    return {model.initial_log_spot, model.initial_variance};
}

__device__ __forceinline__ void one_step_transition(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    std::uint32_t step_index,
    float rough_driver_normal,
    float singular_independent_normal,
    float spot_independent_normal,
    RoughBergomiState& state
) {
    const float brownian_increment =
        model.sqrt_time_step * rough_driver_normal;
    const float orthogonal_increment =
        model.sqrt_time_step * spot_independent_normal;
    const float spot_increment = fmaf(
        model.rho,
        brownian_increment,
        model.orthogonal_correlation * orthogonal_increment
    );

    // Freeze v at the left endpoint and evolve log(S) exactly over this cell.
    state.log_spot = fmaf(
        sqrtf(state.variance),
        spot_increment,
        state.log_spot
            + model.drift_time_step
            - 0.5f * state.variance * model.time_step
    );

    history.brownian_increments[
        static_cast<std::size_t>(step_index) * history.stride
    ] = brownian_increment;

    // kappa=1 integrates the singular current cell exactly. Older cells use
    // the L2-optimal power-kernel constants from the hybrid scheme.
    const float singular_integral = fmaf(
        model.singular_driver_loading,
        rough_driver_normal,
        model.singular_independent_loading * singular_independent_normal
    );
    float far_convolution = 0.0f;
    for (std::uint32_t previous_step = 0U;
         previous_step < step_index;
         ++previous_step) {
        const std::uint32_t weight_index =
            step_index - 1U - previous_step;
        far_convolution = fmaf(
            grid.far_weights[weight_index],
            history.brownian_increments[
                static_cast<std::size_t>(previous_step) * history.stride
            ],
            far_convolution
        );
    }
    const float rough_driver =
        model.sqrt_two_h * (singular_integral + far_convolution);
    state.variance = expf(fmaf(
        model.eta,
        rough_driver,
        grid.log_variance_corrections[step_index]
    ));
}

__device__ __forceinline__ void simulate_one_step(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    std::uint32_t step_index,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    RoughBergomiState& state
) {
    const float rough_driver_normal =
        philox::next_normal(uniforms, normal_cache);
    const float singular_independent_normal =
        philox::next_normal(uniforms, normal_cache);
    const float spot_independent_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        model,
        grid,
        history,
        step_index,
        rough_driver_normal,
        singular_independent_normal,
        spot_independent_normal,
        state
    );
}

__device__ __forceinline__ RoughBergomiState simulate_terminal_state(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    RoughBergomiState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    for (std::uint32_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(
            model, grid, history, step_index, uniforms, normal_cache, state
        );
    }
    return state;
}

__device__ __forceinline__ RoughBergomiMeanPathResult simulate_mean_state(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    RoughBergomiState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    double spot_sum = static_cast<double>(expf(state.log_spot));
    for (std::uint32_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(
            model, grid, history, step_index, uniforms, normal_cache, state
        );
        spot_sum += static_cast<double>(expf(state.log_spot));
    }
    return {
        static_cast<float>(
            spot_sum / (static_cast<double>(num_steps) + 1.0)
        ),
    };
}

__device__ __forceinline__ RoughBergomiGeometricMeanPathResult
simulate_geometric_mean_state(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    RoughBergomiState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    double log_spot_sum = static_cast<double>(state.log_spot);
    for (std::uint32_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(
            model, grid, history, step_index, uniforms, normal_cache, state
        );
        log_spot_sum += static_cast<double>(state.log_spot);
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {
        expf(static_cast<float>(log_spot_sum / observation_count)),
    };
}

__device__ __forceinline__ RoughBergomiTwoTimePathResult
simulate_at_two_times(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t first_step_index,
    std::uint32_t terminal_step_index
) {
    RoughBergomiState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    for (std::uint32_t step_index = 0U;
         step_index < first_step_index;
         ++step_index) {
        simulate_one_step(
            model, grid, history, step_index, uniforms, normal_cache, state
        );
    }
    const float first_spot = expf(state.log_spot);
    for (std::uint32_t step_index = first_step_index;
         step_index < terminal_step_index;
         ++step_index) {
        simulate_one_step(
            model, grid, history, step_index, uniforms, normal_cache, state
        );
    }
    return {first_spot, expf(state.log_spot)};
}

__device__ __forceinline__ RoughBergomiMaximumPathResult
simulate_maximum_state(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    RoughBergomiState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    float maximum_spot = expf(state.log_spot);
    for (std::uint32_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(
            model, grid, history, step_index, uniforms, normal_cache, state
        );
        maximum_spot = fmaxf(maximum_spot, expf(state.log_spot));
    }
    return {maximum_spot};
}

__device__ __forceinline__ RoughBergomiState simulate_on_regular_grid(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_spots
) {
    RoughBergomiState state = initial_state(model);
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    std::uint32_t global_step = 0U;

    for (; global_step < initial_stub_steps; ++global_step) {
        simulate_one_step(
            model, grid, history, global_step, uniforms, normal_cache, state
        );
    }
    if (observation_count == 1U) return state;
    std::size_t output_index = path;
    observed_spots[output_index] = expf(state.log_spot);

    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        const std::uint32_t interval_end =
            global_step + steps_per_observation;
        for (; global_step < interval_end; ++global_step) {
            simulate_one_step(
                model,
                grid,
                history,
                global_step,
                uniforms,
                normal_cache,
                state
            );
        }
        output_index += path_count;
        observed_spots[output_index] = expf(state.log_spot);
    }

    const std::uint32_t terminal_step =
        global_step + steps_per_observation;
    for (; global_step < terminal_step; ++global_step) {
        simulate_one_step(
            model, grid, history, global_step, uniforms, normal_cache, state
        );
    }
    return state;
}

}  // namespace ai_factory::workbench::rough_bergomi
