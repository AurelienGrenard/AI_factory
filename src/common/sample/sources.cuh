// Parameter and calendar sources independent from CUDA execution strategy.
#pragma once

#include "common/check_cuda.cuh"
#include "common/philox.cuh"
#include "common/sample/concepts.cuh"
#include "common/sample/validation.cuh"
#include "common/simulation/schedule.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::sample {

inline constexpr std::uint64_t kParameterSamplingDomain =
    0x6a09e667f3bcc909ULL;
inline constexpr std::uint64_t kScheduleSamplingDomain =
    0xbb67ae8584caa73bULL;

struct MaturityDayBounds {
    std::uint32_t minimum;
    std::uint32_t maximum;
};

struct RandomCalendarDayRules {
    std::uint32_t minimum_observation_day;
    std::uint32_t maximum_observation_day;
    std::uint32_t minimum_interval_days;
};

template<typename Parameters>
struct DeviceParameterSource {
    const Parameters* parameters;

    void validate(std::size_t parameter_count) const {
        if (parameter_count == 0U) {
            throw std::invalid_argument(
                "A parameter source requires at least one row."
            );
        }
        validate_device_pointer(parameters, "device sample parameters");
    }

    __device__ __forceinline__ Parameters load(
        std::size_t parameter_index,
        std::uint64_t
    ) const {
        return parameters[parameter_index];
    }
};

template<ParameterSamplingPolicy Sampler>
struct GeneratedParameterSource {
    using Parameters = typename Sampler::Parameters;
    typename Sampler::Configuration configuration;

    void validate(std::size_t parameter_count) const {
        if (parameter_count == 0U) {
            throw std::invalid_argument(
                "Generated parameters require at least one row."
            );
        }
        Sampler::validate(configuration);
    }

    __device__ __forceinline__ Parameters load(
        std::size_t parameter_index,
        std::uint64_t seed
    ) const {
        return Sampler::sample(
            configuration,
            philox::make_key(seed ^ kParameterSamplingDomain),
            static_cast<std::uint64_t>(parameter_index)
        );
    }
};

template<typename Calendar>
struct ConstantCalendarSource {
    Calendar calendar;

    void validate(std::size_t total_sample_count) const {
        if (total_sample_count == 0U) {
            throw std::invalid_argument(
                "A calendar source requires at least one sample."
            );
        }
        simulation::validate_calendar(calendar);
    }

    template<typename TimeConfiguration>
    void validate(
        std::size_t total_sample_count,
        const TimeConfiguration& time_configuration
    ) const {
        validate(total_sample_count);
        simulation::validate_calendar(calendar, time_configuration);
    }

    __device__ __forceinline__ Calendar load(
        std::size_t,
        std::uint64_t
    ) const {
        return calendar;
    }
};

template<typename Calendar>
struct DeviceCalendarSource {
    const Calendar* host_calendars;
    const Calendar* calendars;

    void validate(std::size_t total_sample_count) const {
        if (total_sample_count == 0U) {
            throw std::invalid_argument(
                "A calendar source requires at least one sample."
            );
        }
        if (host_calendars == nullptr) {
            throw std::invalid_argument(
                "Device sample calendars require a host validation mirror."
            );
        }
        validate_device_pointer(calendars, "device sample calendars");
    }

    template<typename TimeConfiguration>
    void validate(
        std::size_t total_sample_count,
        const TimeConfiguration& time_configuration
    ) const {
        validate(total_sample_count);
        for (std::size_t sample_index = 0U;
             sample_index < total_sample_count;
             ++sample_index) {
            simulation::validate_calendar(
                host_calendars[sample_index],
                time_configuration
            );
        }
    }

    __device__ __forceinline__ Calendar load(
        std::size_t sample_index,
        std::uint64_t
    ) const {
        return calendars[sample_index];
    }
};

// Generate one unbiased integer maturity per logical sample. The optional
// output stores the canonical contractual day used to prepare the schedule.
struct UniformMaturityCalendarSource {
    MaturityDayBounds bounds;
    std::uint32_t* maturity_days;

    void validate(std::size_t total_sample_count) const {
        if (total_sample_count == 0U || bounds.minimum == 0U
            || bounds.minimum > bounds.maximum) {
            throw std::invalid_argument(
                "Maturity-day bounds must be positive and ordered."
            );
        }
        const std::uint64_t support =
            static_cast<std::uint64_t>(bounds.maximum) - bounds.minimum + 1U;
        if (support
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()
            )) {
            throw std::overflow_error(
                "The maturity-day support exceeds uint32_t."
            );
        }
        if (maturity_days != nullptr) {
            validate_device_pointer(
                maturity_days,
                "device generated maturity days"
            );
        }
    }

    template<typename TimeConfiguration>
    void validate(
        std::size_t total_sample_count,
        const TimeConfiguration& time_configuration
    ) const {
        validate(total_sample_count);
        simulation::validate_calendar(
            simulation::MaturityCalendar{bounds.maximum},
            time_configuration
        );
    }

    std::uint32_t maximum_maturity_days() const noexcept {
        return bounds.maximum;
    }

    __device__ __forceinline__ simulation::MaturityCalendar load(
        std::size_t sample_index,
        std::uint64_t seed
    ) const {
        philox::Uint32Sequence integers(
            philox::make_key(seed ^ kScheduleSamplingDomain),
            static_cast<std::uint64_t>(sample_index)
        );
        const std::uint32_t support =
            bounds.maximum - bounds.minimum + 1U;
        const std::uint32_t day = bounds.minimum
            + philox::bounded_uint32(integers, support);
        if (maturity_days != nullptr) maturity_days[sample_index] = day;
        return {day};
    }
};

template<typename Calendar>
concept IntervalDayCalendar = requires(Calendar calendar) {
    calendar.interval_days[0U];
};

// Produce a strictly increasing irregular calendar without materializing
// parallel day/time arrays in thread-local memory. Days are written directly
// in observation-major order when an output view is supplied.
template<typename Calendar, std::size_t ObservationCount>
requires (ObservationCount > 0U && IntervalDayCalendar<Calendar>)
struct RandomIncreasingCalendarSource {
    RandomCalendarDayRules rules;
    std::uint32_t* observation_days;
    std::size_t observation_stride;

    void validate(std::size_t total_sample_count) const {
        if (total_sample_count == 0U
            || rules.minimum_observation_day == 0U
            || rules.minimum_observation_day
                > rules.maximum_observation_day
            || rules.minimum_interval_days == 0U) {
            throw std::invalid_argument(
                "Random-calendar day rules must be positive and ordered."
            );
        }
        const std::uint64_t required_span =
            static_cast<std::uint64_t>(ObservationCount - 1U)
                * rules.minimum_interval_days;
        const std::uint64_t available_span =
            static_cast<std::uint64_t>(rules.maximum_observation_day)
                - rules.minimum_observation_day;
        if (required_span > available_span) {
            throw std::invalid_argument(
                "The random calendar cannot fit inside its day bounds."
            );
        }
        if (observation_days != nullptr) {
            validate_device_pointer(
                observation_days,
                "device generated observation days"
            );
            if (observation_stride < total_sample_count) {
                throw std::invalid_argument(
                    "The observation-day stride is smaller than the sample "
                    "count."
                );
            }
        }
    }

    template<typename TimeConfiguration>
    void validate(
        std::size_t total_sample_count,
        const TimeConfiguration& time_configuration
    ) const {
        validate(total_sample_count);
        simulation::validate_calendar(
            simulation::MaturityCalendar{
                rules.maximum_observation_day
            },
            time_configuration
        );
    }

    std::uint32_t maximum_maturity_days() const noexcept {
        return rules.maximum_observation_day;
    }

    __device__ __forceinline__ Calendar load(
        std::size_t sample_index,
        std::uint64_t seed
    ) const {
        philox::Uint32Sequence integers(
            philox::make_key(seed ^ kScheduleSamplingDomain),
            static_cast<std::uint64_t>(sample_index)
        );
        Calendar calendar{};
        std::uint32_t previous_day = 0U;
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            const std::uint32_t remaining = static_cast<std::uint32_t>(
                ObservationCount - observation - 1U
            );
            const std::uint32_t lower = observation == 0U
                ? rules.minimum_observation_day
                : previous_day + rules.minimum_interval_days;
            const std::uint32_t upper = rules.maximum_observation_day
                - remaining * rules.minimum_interval_days;
            const std::uint32_t day = lower + philox::bounded_uint32(
                integers,
                upper - lower + 1U
            );
            calendar.interval_days[observation] = day - previous_day;
            if (observation_days != nullptr) {
                observation_days[
                    observation * observation_stride + sample_index
                ] = day;
            }
            previous_day = day;
        }
        return calendar;
    }
};

}  // namespace ai_factory::workbench::sample
