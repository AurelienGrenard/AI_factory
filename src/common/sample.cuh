// Shared indexing and host-side validation for model-sample CUDA launchers.
#pragma once

#include "common/check_cuda.cuh"
#include "common/philox.cuh"
#include "common/time_grid.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::sample {

// Independent Philox domains keep conditional parameter generation,
// calendars, and model dynamics reproducible when one of the other stages
// changes its random consumption.
constexpr std::uint64_t kParameterDomain = 0x6a09e667f3bcc909ULL;
constexpr std::uint64_t kScheduleDomain = 0xbb67ae8584caa73bULL;
constexpr std::uint64_t kDynamicsDomain = 0x3c6ef372fe94f82bULL;

// Common closed interval used by every model-specific parameter-bound struct.
struct UniformBounds {
    float minimum;
    float maximum;
};

// A random calendar is generated on the configured integer time grid.  The
// observation count is a launcher/template property so rows have no unused
// trailing storage.
struct RandomCalendarRules {
    UniformBounds observation_time;
    float minimum_observation_interval;
};

struct GeneratedTerminalTime {
    std::uint32_t day;
    float time;
};

template <typename ModelParameters, typename SampleValues>
struct TerminalSampleRow {
    ModelParameters parameters;
    float maturity;
    SampleValues values;
};

template <
    typename ModelParameters,
    typename SampleValues,
    std::uint32_t ObservationCount
>
struct CalendarSampleRow {
    ModelParameters parameters;
    float observation_times[ObservationCount];
    SampleValues values[ObservationCount];
};

__device__ __forceinline__ GeneratedTerminalTime generate_terminal_time(
    philox::Uint32Sequence& integers,
    UniformBounds bounds,
    time_grid::TimeGrid grid
) {
    const std::uint32_t minimum_day = static_cast<std::uint32_t>(
        floorf(bounds.minimum * static_cast<float>(grid.steps_per_year)
            + 0.5f)
    );
    const std::uint32_t maximum_day = static_cast<std::uint32_t>(
        floorf(bounds.maximum * static_cast<float>(grid.steps_per_year)
            + 0.5f)
    );
    const std::uint32_t day_count = maximum_day - minimum_day + 1U;
    const std::uint32_t day = minimum_day
        + philox::bounded_uint32(integers, day_count);
    return {day, time_grid::year_fraction(day, grid)};
}

template <std::uint32_t ObservationCount>
__device__ __forceinline__ void generate_random_calendar(
    philox::Uint32Sequence& integers,
    RandomCalendarRules rules,
    time_grid::TimeGrid grid,
    std::uint32_t (&observation_days)[ObservationCount],
    float (&observation_times)[ObservationCount]
) {
    const std::uint32_t minimum_day = static_cast<std::uint32_t>(
        floorf(rules.observation_time.minimum
                * static_cast<float>(grid.steps_per_year)
            + 0.5f)
    );
    const std::uint32_t maximum_day = static_cast<std::uint32_t>(
        floorf(rules.observation_time.maximum
                * static_cast<float>(grid.steps_per_year)
            + 0.5f)
    );
    const std::uint32_t minimum_gap_days = static_cast<std::uint32_t>(
        floorf(rules.minimum_observation_interval
                * static_cast<float>(grid.steps_per_year)
            + 0.5f)
    );

    std::uint32_t previous_day = 0U;
    #pragma unroll
    for (std::uint32_t observation = 0U;
         observation < ObservationCount;
         ++observation) {
        const std::uint32_t remaining_intervals =
            ObservationCount - observation - 1U;
        const std::uint32_t lower_day = observation == 0U
            ? minimum_day
            : previous_day + minimum_gap_days;
        const std::uint32_t upper_day = maximum_day
            - remaining_intervals * minimum_gap_days;
        const std::uint32_t day = lower_day + philox::bounded_uint32(
            integers, upper_day - lower_day + 1U
        );
        observation_days[observation] = day;
        observation_times[observation] = time_grid::year_fraction(day, grid);
        previous_day = day;
    }
}

inline void validate_uniform_bounds(
    UniformBounds bounds,
    const char* name
) {
    if (!std::isfinite(bounds.minimum)
        || !std::isfinite(bounds.maximum)
        || bounds.minimum > bounds.maximum) {
        throw std::invalid_argument(
            std::string(name) + " bounds must be finite and ordered."
        );
    }
}

inline void validate_terminal_bounds(
    UniformBounds maturity,
    time_grid::TimeGrid grid
) {
    time_grid::validate(grid);
    validate_uniform_bounds(maturity, "Terminal maturity");
    const std::uint32_t minimum_day = time_grid::index(
        maturity.minimum, grid, "Minimum terminal maturity"
    );
    const std::uint32_t maximum_day = time_grid::index(
        maturity.maximum, grid, "Maximum terminal maturity"
    );
    if (minimum_day > maximum_day) {
        throw std::invalid_argument(
            "Terminal maturity day bounds must be ordered."
        );
    }
}

inline void validate_generated_sample_launch(
    std::size_t total_sample_count,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    const void* device_rows
) {
    validate_device_pointer(device_rows, "device_rows");
    if (total_sample_count == 0U || paths_per_parameter == 0U) {
        throw std::invalid_argument(
            "Sample and paths-per-parameter counts must be positive."
        );
    }
    if (total_sample_count % paths_per_parameter != 0U) {
        throw std::invalid_argument(
            "The total sample count must contain complete parameter packages."
        );
    }
    if (sample_offset >= total_sample_count
        || launch_sample_count == 0U
        || launch_sample_count > total_sample_count - sample_offset) {
        throw std::invalid_argument(
            "The generated-sample launch batch exceeds the output array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    if (paths_per_parameter > 1U
        && paths_per_parameter > threads_per_block) {
        throw std::invalid_argument(
            "A packaged sample launch requires at least one thread per path."
        );
    }
    const std::size_t first_package = sample_offset / paths_per_parameter;
    const std::size_t final_sample = sample_offset + launch_sample_count - 1U;
    const std::size_t final_package = final_sample / paths_per_parameter;
    const std::size_t launch_package_count =
        final_package - first_package + 1U;
    validate_block_count(
        paths_per_parameter == 1U
            ? launch_sample_count
            : launch_package_count,
        block_count
    );
    validate_grid_x_size(block_count);
    if (total_sample_count
        > static_cast<std::size_t>(
            std::numeric_limits<std::uint64_t>::max()
        )) {
        throw std::overflow_error(
            "Sample indices exceed the Philox uint64 path domain."
        );
    }
}

template <std::uint32_t ObservationCount>
inline void validate_random_calendar_rules(
    RandomCalendarRules rules,
    time_grid::TimeGrid grid
) {
    static_assert(ObservationCount > 0U);
    time_grid::validate(grid);
    validate_uniform_bounds(
        rules.observation_time, "Random-calendar observation time"
    );
    const std::uint32_t minimum_day = time_grid::index(
        rules.observation_time.minimum,
        grid,
        "Minimum random-calendar observation time"
    );
    const std::uint32_t maximum_day = time_grid::index(
        rules.observation_time.maximum,
        grid,
        "Maximum random-calendar observation time"
    );
    const std::uint32_t minimum_gap_days = time_grid::index(
        rules.minimum_observation_interval,
        grid,
        "Minimum random-calendar observation interval"
    );
    const std::uint64_t required_span =
        static_cast<std::uint64_t>(ObservationCount - 1U)
            * minimum_gap_days;
    if (required_span
        > static_cast<std::uint64_t>(maximum_day - minimum_day)) {
        throw std::invalid_argument(
            "The requested random calendar cannot fit inside its time bounds."
        );
    }
}

template <std::uint32_t ObservationCount>
inline void validate_fixed_calendar(
    const float* observation_times,
    time_grid::TimeGrid grid
) {
    time_grid::validate(grid);
    if (observation_times == nullptr) {
        throw std::invalid_argument(
            "Fixed-calendar observation times must not be null."
        );
    }
    std::uint32_t previous_day = 0U;
    for (std::uint32_t observation = 0U;
         observation < ObservationCount;
         ++observation) {
        const std::uint32_t day = time_grid::index(
            observation_times[observation],
            grid,
            "Fixed-calendar observation time"
        );
        if (observation != 0U && day <= previous_day) {
            throw std::invalid_argument(
                "Fixed-calendar observation times must be strictly increasing."
            );
        }
        previous_day = day;
    }
}

// Decode one flattened sample into its model row and conditional path index.
struct ModelPathIndices {
    std::size_t model_index;
    std::size_t path_index;
};

__host__ __device__ constexpr ModelPathIndices decode_sample_index(
    std::size_t sample_index,
    std::size_t paths_per_model
) noexcept {
    return {
        sample_index / paths_per_model,
        sample_index % paths_per_model,
    };
}

// Compute M * P without accepting empty datasets or size_t overflow.
inline std::size_t sample_count(
    std::size_t model_count,
    std::size_t paths_per_model
) {
    if (model_count == 0U || paths_per_model == 0U) {
        throw std::invalid_argument(
            "Model count and paths per model must be positive."
        );
    }
    if (model_count
        > std::numeric_limits<std::size_t>::max() / paths_per_model) {
        throw std::overflow_error("Model sample count exceeds size_t.");
    }
    return model_count * paths_per_model;
}

// Validate the common arrays, sample slice, persistent grid, and Philox keys.
inline std::size_t validate_sample_launch(
    const void* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    const void* device_primary_samples
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_primary_samples, "device_primary_samples");
    const std::size_t total_sample_count = sample_count(
        model_count, paths_per_model
    );
    if (sample_offset >= total_sample_count
        || launch_sample_count == 0U
        || launch_sample_count > total_sample_count - sample_offset) {
        throw std::invalid_argument(
            "The sample launch batch exceeds the output arrays."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_sample_count, block_count);
    validate_grid_x_size(block_count);
    validate_row_seed_range(model_count, base_seed);
    return total_sample_count;
}

// Require one finite positive terminal horizon.
inline void validate_terminal_time(float maturity) {
    if (!(maturity > 0.0f) || !std::isfinite(maturity)) {
        throw std::invalid_argument(
            "Sample maturity must be positive and finite."
        );
    }
}

// Require a finite, positive, regular observation calendar.
inline void validate_regular_calendar(
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count
) {
    if (!(first_observation_time > 0.0f)
        || !std::isfinite(first_observation_time)
        || !(observation_interval > 0.0f)
        || !std::isfinite(observation_interval)
        || observation_count == 0U) {
        throw std::invalid_argument(
            "A regular sample calendar requires positive finite times and "
            "at least one observation."
        );
    }
}

// Convert one discretized interval to its nearest positive number of steps.
inline std::uint32_t rounded_step_count(float interval, float target_dt) {
    if (!(interval > 0.0f) || !std::isfinite(interval)
        || !(target_dt > 0.0f) || !std::isfinite(target_dt)) {
        throw std::invalid_argument(
            "Discretized sampling requires positive finite times."
        );
    }
    const double rounded = std::floor(
        static_cast<double>(interval) / static_cast<double>(target_dt) + 0.5
    );
    if (!(rounded >= 1.0)
        || rounded > static_cast<double>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::overflow_error(
            "The requested sample step count is invalid or too large."
        );
    }
    return static_cast<std::uint32_t>(rounded);
}

// Validate that a calendar's final step index fits uint32_t.
inline std::uint32_t calendar_step_count(
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count
) {
    if (observation_count == 0U) return 0U;
    const std::uint64_t total = initial_stub_steps
        + static_cast<std::uint64_t>(observation_count - 1U)
            * steps_per_observation;
    if (total > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "The sample calendar step count exceeds uint32_t."
        );
    }
    return static_cast<std::uint32_t>(total);
}

}  // namespace ai_factory::workbench::sample
