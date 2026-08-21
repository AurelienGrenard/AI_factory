// Persistent Black-Scholes kernels generating parameters, schedules, and rows.
#include "model/equity/black_scholes/sample.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/sample.cuh"

// Include the reusable dynamics so NVCC can inline every transition.
#include "model/equity/black_scholes/dynamics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::black_scholes {
namespace {

template <std::uint32_t ObservationCount>
struct FixedCalendar {
    std::uint32_t days[ObservationCount];
    float times[ObservationCount];
};

__device__ __forceinline__ float generate_uniform(
    sample::UniformBounds bounds,
    philox::UniformSequence& uniforms
) {
    return fmaf(
        bounds.maximum - bounds.minimum,
        uniforms.next(),
        bounds.minimum
    );
}

// This model-specific function deliberately has the same role and signature
// as the adjacent sample implementation in every other model.
__device__ __forceinline__ ModelParameters generate_parameters(
    BlackScholesSampleBounds bounds,
    std::uint64_t base_seed,
    std::uint64_t parameter_index
) {
    philox::UniformSequence uniforms(
        philox::make_key(base_seed ^ sample::kParameterDomain),
        parameter_index
    );
    return {
        generate_uniform(bounds.spot, uniforms),
        generate_uniform(bounds.risk_free_rate, uniforms),
        generate_uniform(bounds.dividend_yield, uniforms),
        generate_uniform(bounds.volatility, uniforms),
    };
}

// Isolate the one-thread parameter draw from the packaged simulation's
// register allocation.  One device call per 250 paths is negligible compared
// with making every path reserve the Philox-generation registers.
__device__ __noinline__ void generate_shared_parameters(
    BlackScholesSampleBounds bounds,
    std::uint64_t base_seed,
    std::uint64_t parameter_index,
    ModelParameters* shared_parameters
) {
    *shared_parameters = generate_parameters(
        bounds, base_seed, parameter_index
    );
}

__device__ __forceinline__ void write_terminal_sample(
    ModelParameters parameters,
    sample::UniformBounds maturity_bounds,
    time_grid::TimeGrid grid,
    std::size_t sample_index,
    std::uint64_t base_seed,
    BlackScholesTerminalSampleRow* rows
) {
    philox::Uint32Sequence schedule_integers(
        philox::make_key(base_seed ^ sample::kScheduleDomain),
        static_cast<std::uint64_t>(sample_index)
    );
    const sample::GeneratedTerminalTime maturity =
        sample::generate_terminal_time(
            schedule_integers, maturity_bounds, grid
        );
    const PreparedModel prepared = prepare_model(parameters);
    const PreparedTransition transition = prepare_transition(
        prepared, maturity.time
    );
    const State terminal = simulate_terminal_state(
        prepared,
        transition,
        philox::make_key(base_seed ^ sample::kDynamicsDomain),
        sample_index
    );
    const BlackScholesTerminalSampleRow row{
        parameters,
        maturity.time,
        {expf(terminal.log_spot)},
    };
    rows[sample_index] = row;
}

template <bool Packaged>
__global__ void black_scholes_terminal_samples_kernel(
    BlackScholesSampleBounds parameter_bounds,
    sample::UniformBounds maturity_bounds,
    time_grid::TimeGrid grid,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    BlackScholesTerminalSampleRow* __restrict__ rows
) {
    if constexpr (!Packaged) {
        const std::size_t thread =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride =
            static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t launch_index = thread;
             launch_index < launch_sample_count;
             launch_index += stride) {
            const std::size_t sample_index = sample_offset + launch_index;
            const ModelParameters parameters =
                generate_parameters(
                    parameter_bounds, base_seed, sample_index
                );
            write_terminal_sample(
                parameters, maturity_bounds, grid, sample_index, base_seed,
                rows
            );
        }
    } else {
        __shared__ ModelParameters shared_parameters;
        const std::size_t first_package =
            sample_offset / paths_per_parameter;
        const std::size_t final_sample =
            sample_offset + launch_sample_count - 1U;
        const std::size_t final_package =
            final_sample / paths_per_parameter;
        for (std::size_t parameter_index = first_package + blockIdx.x;
             parameter_index <= final_package;
            parameter_index += gridDim.x) {
            if (threadIdx.x == 0U) {
                generate_shared_parameters(
                    parameter_bounds,
                    base_seed,
                    parameter_index,
                    &shared_parameters
                );
            }
            __syncthreads();
            const std::size_t sample_index =
                parameter_index * paths_per_parameter + threadIdx.x;
            if (threadIdx.x < paths_per_parameter
                && sample_index >= sample_offset
                && sample_index <= final_sample) {
                const ModelParameters parameters =
                    shared_parameters;
                write_terminal_sample(
                    parameters,
                    maturity_bounds,
                    grid,
                    sample_index,
                    base_seed,
                    rows
                );
            }
            __syncthreads();
        }
    }
}

template <std::uint32_t ObservationCount, bool RandomCalendar>
__device__ __forceinline__ void write_calendar_sample(
    ModelParameters parameters,
    sample::RandomCalendarRules random_rules,
    FixedCalendar<ObservationCount> fixed_calendar,
    time_grid::TimeGrid grid,
    std::size_t sample_index,
    std::uint64_t base_seed,
    BlackScholesCalendarSampleRow<ObservationCount>* rows
) {
    std::uint32_t observation_days[ObservationCount];
    float observation_times[ObservationCount];
    if constexpr (RandomCalendar) {
        philox::Uint32Sequence schedule_integers(
            philox::make_key(base_seed ^ sample::kScheduleDomain),
            static_cast<std::uint64_t>(sample_index)
        );
        sample::generate_random_calendar(
            schedule_integers,
            random_rules,
            grid,
            observation_days,
            observation_times
        );
    } else {
        #pragma unroll
        for (std::uint32_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            observation_days[observation] = fixed_calendar.days[observation];
            observation_times[observation] = fixed_calendar.times[observation];
        }
    }

    std::uint32_t interval_steps[ObservationCount];
    std::uint32_t previous_day = 0U;
    #pragma unroll
    for (std::uint32_t observation = 0U;
         observation < ObservationCount;
         ++observation) {
        interval_steps[observation] =
            observation_days[observation] - previous_day;
        previous_day = observation_days[observation];
    }
    const PreparedModel prepared = prepare_model(parameters);
    PreparedTransition transitions[ObservationCount];
    prepare_calendar(
        prepared,
        interval_steps,
        ObservationCount,
        grid.step_size,
        transitions
    );
    float observed_spots[ObservationCount];
    const State terminal = simulate_on_calendar(
        prepared,
        transitions,
        ObservationCount,
        philox::make_key(base_seed ^ sample::kDynamicsDomain),
        sample_index,
        1U,
        observed_spots
    );
    observed_spots[ObservationCount - 1U] = expf(terminal.log_spot);

    BlackScholesCalendarSampleRow<ObservationCount> row{};
    row.parameters = parameters;
    #pragma unroll
    for (std::uint32_t observation = 0U;
         observation < ObservationCount;
         ++observation) {
        row.observation_times[observation] = observation_times[observation];
        row.values[observation] = {observed_spots[observation]};
    }
    rows[sample_index] = row;
}

template <
    std::uint32_t ObservationCount,
    bool RandomCalendar,
    bool Packaged
>
__global__ void black_scholes_calendar_samples_kernel(
    BlackScholesSampleBounds parameter_bounds,
    sample::RandomCalendarRules random_rules,
    FixedCalendar<ObservationCount> fixed_calendar,
    time_grid::TimeGrid grid,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    BlackScholesCalendarSampleRow<ObservationCount>* __restrict__ rows
) {
    if constexpr (!Packaged) {
        const std::size_t thread =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride =
            static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t launch_index = thread;
             launch_index < launch_sample_count;
             launch_index += stride) {
            const std::size_t sample_index = sample_offset + launch_index;
            const ModelParameters parameters =
                generate_parameters(
                    parameter_bounds, base_seed, sample_index
                );
            write_calendar_sample<ObservationCount, RandomCalendar>(
                parameters,
                random_rules,
                fixed_calendar,
                grid,
                sample_index,
                base_seed,
                rows
            );
        }
    } else {
        __shared__ ModelParameters shared_parameters;
        const std::size_t first_package =
            sample_offset / paths_per_parameter;
        const std::size_t final_sample =
            sample_offset + launch_sample_count - 1U;
        const std::size_t final_package =
            final_sample / paths_per_parameter;
        for (std::size_t parameter_index = first_package + blockIdx.x;
             parameter_index <= final_package;
            parameter_index += gridDim.x) {
            if (threadIdx.x == 0U) {
                generate_shared_parameters(
                    parameter_bounds,
                    base_seed,
                    parameter_index,
                    &shared_parameters
                );
            }
            __syncthreads();
            const std::size_t sample_index =
                parameter_index * paths_per_parameter + threadIdx.x;
            if (threadIdx.x < paths_per_parameter
                && sample_index >= sample_offset
                && sample_index <= final_sample) {
                const ModelParameters parameters =
                    shared_parameters;
                write_calendar_sample<ObservationCount, RandomCalendar>(
                    parameters,
                    random_rules,
                    fixed_calendar,
                    grid,
                    sample_index,
                    base_seed,
                    rows
                );
            }
            __syncthreads();
        }
    }
}

void validate_parameter_bounds(BlackScholesSampleBounds bounds) {
    sample::validate_uniform_bounds(bounds.spot, "Black-Scholes spot");
    sample::validate_uniform_bounds(
        bounds.risk_free_rate, "Black-Scholes risk-free rate"
    );
    sample::validate_uniform_bounds(
        bounds.dividend_yield, "Black-Scholes dividend yield"
    );
    sample::validate_uniform_bounds(
        bounds.volatility, "Black-Scholes volatility"
    );
    if (!(bounds.spot.minimum > 0.0f)
        || !(bounds.volatility.minimum > 0.0f)) {
        throw std::invalid_argument(
            "Black-Scholes spot and volatility bounds must be positive."
        );
    }
}

template <std::uint32_t ObservationCount, bool RandomCalendar>
void launch_calendar(
    BlackScholesSampleBounds parameter_bounds,
    sample::RandomCalendarRules random_rules,
    FixedCalendar<ObservationCount> fixed_calendar,
    time_grid::TimeGrid time_grid_config,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    void* device_rows
) {
    using Row = BlackScholesCalendarSampleRow<ObservationCount>;
    Row* rows = static_cast<Row*>(device_rows);
    const dim3 grid(static_cast<unsigned int>(block_count));
    const dim3 block(threads_per_block);
    if (paths_per_parameter == 1U) {
        report_cuda_kernel_launch_if_enabled(
            "black_scholes.samples",
            RandomCalendar ? "random_calendar.iid" : "fixed_calendar.iid",
            black_scholes_calendar_samples_kernel<
                ObservationCount, RandomCalendar, false
            >,
            grid,
            block
        );
        black_scholes_calendar_samples_kernel<
            ObservationCount, RandomCalendar, false
        ><<<grid, block>>>(
            parameter_bounds,
            random_rules,
            fixed_calendar,
            time_grid_config,
            paths_per_parameter,
            sample_offset,
            launch_sample_count,
            base_seed,
            rows
        );
    } else {
        report_cuda_kernel_launch_if_enabled(
            "black_scholes.samples",
            RandomCalendar
                ? "random_calendar.packaged"
                : "fixed_calendar.packaged",
            black_scholes_calendar_samples_kernel<
                ObservationCount, RandomCalendar, true
            >,
            grid,
            block
        );
        black_scholes_calendar_samples_kernel<
            ObservationCount, RandomCalendar, true
        ><<<grid, block>>>(
            parameter_bounds,
            random_rules,
            fixed_calendar,
            time_grid_config,
            paths_per_parameter,
            sample_offset,
            launch_sample_count,
            base_seed,
            rows
        );
    }
    check_cuda(
        cudaGetLastError(), "Black-Scholes calendar sample kernel"
    );
}

template <std::uint32_t ObservationCount>
void launch_random_calendar(
    BlackScholesSampleBounds parameter_bounds,
    sample::RandomCalendarRules calendar_rules,
    time_grid::TimeGrid time_grid_config,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    void* device_rows
) {
    sample::validate_random_calendar_rules<ObservationCount>(
        calendar_rules, time_grid_config
    );
    launch_calendar<ObservationCount, true>(
        parameter_bounds,
        calendar_rules,
        {},
        time_grid_config,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
        threads_per_block,
        block_count,
        base_seed,
        device_rows
    );
}

template <std::uint32_t ObservationCount>
void launch_fixed_calendar(
    BlackScholesSampleBounds parameter_bounds,
    const float* observation_times,
    time_grid::TimeGrid time_grid_config,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    void* device_rows
) {
    sample::validate_fixed_calendar<ObservationCount>(
        observation_times, time_grid_config
    );
    FixedCalendar<ObservationCount> calendar{};
    for (std::uint32_t observation = 0U;
         observation < ObservationCount;
         ++observation) {
        const std::uint32_t day = time_grid::index(
            observation_times[observation],
            time_grid_config,
            "Black-Scholes fixed-calendar observation time"
        );
        calendar.days[observation] = day;
        calendar.times[observation] = time_grid::year_fraction(
            day, time_grid_config
        );
    }
    launch_calendar<ObservationCount, false>(
        parameter_bounds,
        {},
        calendar,
        time_grid_config,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
        threads_per_block,
        block_count,
        base_seed,
        device_rows
    );
}

#define AI_FACTORY_DISPATCH_RANDOM_CALENDAR(COUNT)                         \
    case COUNT:                                                            \
        launch_random_calendar<COUNT>(                                     \
            parameter_bounds, calendar_rules, time_grid_config,           \
            paths_per_parameter,                                          \
            sample_offset, launch_sample_count, threads_per_block,         \
            block_count, base_seed, device_rows                            \
        );                                                                 \
        return

#define AI_FACTORY_DISPATCH_FIXED_CALENDAR(COUNT)                          \
    case COUNT:                                                            \
        launch_fixed_calendar<COUNT>(                                      \
            parameter_bounds, observation_times, time_grid_config,        \
            paths_per_parameter,                                          \
            sample_offset, launch_sample_count, threads_per_block,         \
            block_count, base_seed, device_rows                            \
        );                                                                 \
        return

}  // namespace

void launch_black_scholes_terminal_samples_cuda(
    BlackScholesSampleBounds parameter_bounds,
    sample::UniformBounds maturity_bounds,
    time_grid::TimeGrid time_grid_config,
    std::size_t total_sample_count,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    BlackScholesTerminalSampleRow* device_rows
) {
    validate_parameter_bounds(parameter_bounds);
    sample::validate_terminal_bounds(maturity_bounds, time_grid_config);
    sample::validate_generated_sample_launch(
        total_sample_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
        threads_per_block,
        block_count,
        device_rows
    );
    const dim3 grid(static_cast<unsigned int>(block_count));
    const dim3 block(threads_per_block);
    if (paths_per_parameter == 1U) {
        report_cuda_kernel_launch_if_enabled(
            "black_scholes.samples",
            "terminal.iid",
            black_scholes_terminal_samples_kernel<false>,
            grid,
            block
        );
        black_scholes_terminal_samples_kernel<false><<<grid, block>>>(
            parameter_bounds,
            maturity_bounds,
            time_grid_config,
            paths_per_parameter,
            sample_offset,
            launch_sample_count,
            base_seed,
            device_rows
        );
    } else {
        report_cuda_kernel_launch_if_enabled(
            "black_scholes.samples",
            "terminal.packaged",
            black_scholes_terminal_samples_kernel<true>,
            grid,
            block
        );
        black_scholes_terminal_samples_kernel<true><<<grid, block>>>(
            parameter_bounds,
            maturity_bounds,
            time_grid_config,
            paths_per_parameter,
            sample_offset,
            launch_sample_count,
            base_seed,
            device_rows
        );
    }
    check_cuda(
        cudaGetLastError(), "Black-Scholes terminal sample kernel"
    );
}

void launch_black_scholes_random_calendar_samples_cuda(
    BlackScholesSampleBounds parameter_bounds,
    sample::RandomCalendarRules calendar_rules,
    time_grid::TimeGrid time_grid_config,
    std::uint32_t observation_count,
    std::size_t total_sample_count,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    void* device_rows
) {
    validate_parameter_bounds(parameter_bounds);
    sample::validate_generated_sample_launch(
        total_sample_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
        threads_per_block,
        block_count,
        device_rows
    );
    switch (observation_count) {
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(1U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(2U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(3U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(4U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(5U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(6U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(7U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(8U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(9U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(10U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(11U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(12U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(13U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(14U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(15U);
        AI_FACTORY_DISPATCH_RANDOM_CALENDAR(16U);
        default:
            throw std::invalid_argument(
                "Black-Scholes sample calendars support 1 to 16 observations."
            );
    }
}

void launch_black_scholes_fixed_calendar_samples_cuda(
    BlackScholesSampleBounds parameter_bounds,
    const float* observation_times,
    time_grid::TimeGrid time_grid_config,
    std::uint32_t observation_count,
    std::size_t total_sample_count,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    void* device_rows
) {
    validate_parameter_bounds(parameter_bounds);
    sample::validate_generated_sample_launch(
        total_sample_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
        threads_per_block,
        block_count,
        device_rows
    );
    switch (observation_count) {
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(1U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(2U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(3U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(4U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(5U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(6U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(7U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(8U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(9U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(10U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(11U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(12U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(13U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(14U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(15U);
        AI_FACTORY_DISPATCH_FIXED_CALENDAR(16U);
        default:
            throw std::invalid_argument(
                "Black-Scholes sample calendars support 1 to 16 observations."
            );
    }
}

#undef AI_FACTORY_DISPATCH_RANDOM_CALENDAR
#undef AI_FACTORY_DISPATCH_FIXED_CALENDAR

}  // namespace ai_factory::workbench::black_scholes
