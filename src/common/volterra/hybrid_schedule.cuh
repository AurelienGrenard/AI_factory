// Calendar-to-grid policies for block-cooperative Volterra path executors.
#pragma once

#include "common/check_cuda.cuh"
#include "common/simulation/schedule.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::volterra {

struct HybridTimeConfiguration {
    float day_fraction;
    float target_dt;
};

inline void validate_time_configuration(
    const HybridTimeConfiguration& time_configuration
) {
    validate_day_fraction(time_configuration.day_fraction);
    validate_time_step(time_configuration.target_dt);
}

inline std::uint32_t rounded_step_count(
    std::uint32_t maturity_days,
    const HybridTimeConfiguration& time_configuration
) {
    validate_time_configuration(time_configuration);
    if (maturity_days == 0U) {
        throw std::invalid_argument(
            "A Volterra hybrid schedule requires a positive maturity."
        );
    }
    const double maturity = static_cast<double>(maturity_days)
        * static_cast<double>(time_configuration.day_fraction);
    const double rounded = std::max(
        1.0,
        std::floor(
            maturity / static_cast<double>(time_configuration.target_dt)
            + 0.5
        )
    );
    if (rounded > static_cast<double>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::overflow_error(
            "The Volterra hybrid schedule step count exceeds uint32_t."
        );
    }
    return static_cast<std::uint32_t>(rounded);
}

__host__ __device__ inline std::uint32_t rounded_observation_step(
    std::uint64_t cumulative_days,
    std::uint64_t maturity_days,
    std::uint32_t step_count
) {
    const std::uint64_t numerator = cumulative_days * step_count;
    return static_cast<std::uint32_t>(
        (numerator + maturity_days / 2U) / maturity_days
    );
}

struct TerminalHybridSchedule {
    using Calendar = simulation::MaturityCalendar;

    struct PreparedSchedule {
        float maturity_years;
        float time_step;
        std::uint32_t step_count;
    };

    struct Cursor {};

    static std::uint32_t execution_step_count(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration
    ) {
        return rounded_step_count(
            calendar.maturity_days,
            time_configuration
        );
    }

    __device__ __forceinline__ static PreparedSchedule prepare(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration,
        std::uint32_t step_count
    ) {
        const float maturity = static_cast<float>(calendar.maturity_days)
            * time_configuration.day_fraction;
        return {
            maturity,
            maturity / static_cast<float>(step_count),
            step_count,
        };
    }

    __device__ __forceinline__ static Cursor make_cursor(
        const PreparedSchedule&
    ) {
        return {};
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_initial_state(
        const PreparedSchedule&,
        Cursor&,
        const State&,
        Handler&
    ) {
        return true;
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_step(
        const PreparedSchedule& schedule,
        Cursor&,
        std::uint32_t step,
        const State& state,
        Handler& handler
    ) {
        return step + 1U != schedule.step_count
            || handler.on_observation(0U, state);
    }
};

struct DenseHybridSchedule {
    using Calendar = simulation::MaturityCalendar;
    using PreparedSchedule = TerminalHybridSchedule::PreparedSchedule;
    struct Cursor {};

    static std::uint32_t execution_step_count(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration
    ) {
        return rounded_step_count(
            calendar.maturity_days,
            time_configuration
        );
    }

    __device__ __forceinline__ static PreparedSchedule prepare(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration,
        std::uint32_t step_count
    ) {
        return TerminalHybridSchedule::prepare(
            calendar,
            time_configuration,
            step_count
        );
    }

    __device__ __forceinline__ static Cursor make_cursor(
        const PreparedSchedule&
    ) {
        return {};
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_initial_state(
        const PreparedSchedule&,
        Cursor&,
        const State& state,
        Handler& handler
    ) {
        return handler.on_initial_state(state);
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_step(
        const PreparedSchedule&,
        Cursor&,
        std::uint32_t step,
        const State& state,
        Handler& handler
    ) {
        return handler.on_observation(step, state);
    }
};

struct RegularHybridSchedule {
    using Calendar = simulation::RegularCalendar;

    struct PreparedSchedule {
        float maturity_years;
        float time_step;
        std::uint32_t step_count;
        std::uint32_t observation_count;
    };

    struct Cursor {
        std::uint32_t observation;
    };

    static std::uint32_t execution_step_count(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration
    ) {
        if (calendar.observation_interval_days == 0U
            || calendar.observation_count == 0U) {
            throw std::invalid_argument(
                "A regular Volterra schedule requires a positive interval "
                "and observation count."
            );
        }
        const std::uint64_t days =
            static_cast<std::uint64_t>(calendar.observation_interval_days)
            * calendar.observation_count;
        if (days > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "The regular Volterra schedule maturity exceeds uint32_t."
            );
        }
        return rounded_step_count(
            static_cast<std::uint32_t>(days),
            time_configuration
        );
    }

    __device__ __forceinline__ static PreparedSchedule prepare(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration,
        std::uint32_t step_count
    ) {
        const std::uint64_t maturity_days =
            static_cast<std::uint64_t>(calendar.observation_interval_days)
            * calendar.observation_count;
        const float maturity = static_cast<float>(maturity_days)
            * time_configuration.day_fraction;
        return {
            maturity,
            maturity / static_cast<float>(step_count),
            step_count,
            calendar.observation_count,
        };
    }

    __device__ __forceinline__ static Cursor make_cursor(
        const PreparedSchedule&
    ) {
        return {0U};
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_initial_state(
        const PreparedSchedule&,
        Cursor&,
        const State& state,
        Handler& handler
    ) {
        return handler.on_initial_state(state);
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_step(
        const PreparedSchedule& schedule,
        Cursor& cursor,
        std::uint32_t step,
        const State& state,
        Handler& handler
    ) {
        bool keep_running = true;
        while (keep_running
               && cursor.observation < schedule.observation_count
               && rounded_observation_step(
                    cursor.observation + 1U,
                    schedule.observation_count,
                    schedule.step_count
               ) == step + 1U) {
            keep_running = handler.on_observation(
                cursor.observation,
                state
            );
            ++cursor.observation;
        }
        return keep_running;
    }
};

struct StubbedRegularHybridSchedule {
    using Calendar = simulation::StubbedRegularCalendar;

    struct PreparedSchedule {
        float maturity_years;
        float time_step;
        std::uint32_t step_count;
        std::uint32_t first_observation_step;
        std::uint32_t observation_interval_steps;
        std::uint32_t observation_count;
    };

    struct Cursor {
        std::uint32_t observation;
    };

    static std::uint32_t execution_step_count(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration
    ) {
        simulation::validate_calendar(calendar);
        const std::uint64_t maturity_days = calendar.first_observation_day
            + static_cast<std::uint64_t>(calendar.observation_count - 1U)
                * calendar.observation_interval_days;
        if (maturity_days > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "The stubbed Volterra schedule maturity exceeds uint32_t."
            );
        }
        return rounded_step_count(
            static_cast<std::uint32_t>(maturity_days),
            time_configuration
        );
    }

    __device__ __forceinline__ static PreparedSchedule prepare(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration,
        std::uint32_t step_count
    ) {
        const std::uint64_t maturity_days = calendar.first_observation_day
            + static_cast<std::uint64_t>(calendar.observation_count - 1U)
                * calendar.observation_interval_days;
        const float maturity = static_cast<float>(maturity_days)
            * time_configuration.day_fraction;
        return {
            maturity,
            maturity / static_cast<float>(step_count),
            step_count,
            rounded_observation_step(
                calendar.first_observation_day,
                maturity_days,
                step_count
            ),
            rounded_observation_step(
                calendar.observation_interval_days,
                maturity_days,
                step_count
            ),
            calendar.observation_count,
        };
    }

    __device__ __forceinline__ static Cursor make_cursor(
        const PreparedSchedule&
    ) {
        return {0U};
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_initial_state(
        const PreparedSchedule&,
        Cursor&,
        const State& state,
        Handler& handler
    ) {
        return handler.on_initial_state(state);
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_step(
        const PreparedSchedule& schedule,
        Cursor& cursor,
        std::uint32_t step,
        const State& state,
        Handler& handler
    ) {
        bool keep_running = true;
        while (keep_running
               && cursor.observation < schedule.observation_count
               && schedule.first_observation_step
                    + cursor.observation
                        * schedule.observation_interval_steps
                    == step + 1U) {
            keep_running = handler.on_observation(
                cursor.observation,
                state
            );
            ++cursor.observation;
        }
        return keep_running;
    }
};

template<std::size_t ObservationCount>
requires (ObservationCount > 0U)
struct CalendarHybridSchedule {
    using Calendar = simulation::StaticCalendar<ObservationCount>;

    struct PreparedSchedule {
        float maturity_years;
        float time_step;
        std::uint32_t step_count;
        std::uint32_t observation_steps[ObservationCount];
    };

    struct Cursor {
        std::uint32_t observation;
    };

    static std::uint32_t execution_step_count(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration
    ) {
        std::uint64_t maturity_days = 0U;
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            if (calendar.interval_days[observation] == 0U) {
                throw std::invalid_argument(
                    "A Volterra calendar requires positive intervals."
                );
            }
            maturity_days += calendar.interval_days[observation];
        }
        if (maturity_days > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "The Volterra calendar maturity exceeds uint32_t."
            );
        }
        return rounded_step_count(
            static_cast<std::uint32_t>(maturity_days),
            time_configuration
        );
    }

    __device__ __forceinline__ static PreparedSchedule prepare(
        const Calendar& calendar,
        const HybridTimeConfiguration& time_configuration,
        std::uint32_t step_count
    ) {
        PreparedSchedule schedule{};
        std::uint64_t maturity_days = 0U;
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            maturity_days += calendar.interval_days[observation];
        }
        schedule.maturity_years = static_cast<float>(maturity_days)
            * time_configuration.day_fraction;
        schedule.time_step =
            schedule.maturity_years / static_cast<float>(step_count);
        schedule.step_count = step_count;
        std::uint64_t cumulative_days = 0U;
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            cumulative_days += calendar.interval_days[observation];
            schedule.observation_steps[observation] =
                rounded_observation_step(
                    cumulative_days,
                    maturity_days,
                    step_count
                );
        }
        return schedule;
    }

    __device__ __forceinline__ static Cursor make_cursor(
        const PreparedSchedule&
    ) {
        return {0U};
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_initial_state(
        const PreparedSchedule&,
        Cursor&,
        const State& state,
        Handler& handler
    ) {
        return handler.on_initial_state(state);
    }

    template<typename State, typename Handler>
    __device__ __forceinline__ static bool on_step(
        const PreparedSchedule& schedule,
        Cursor& cursor,
        std::uint32_t step,
        const State& state,
        Handler& handler
    ) {
        bool keep_running = true;
        while (keep_running
               && cursor.observation < ObservationCount
               && schedule.observation_steps[cursor.observation]
                    == step + 1U) {
            keep_running = handler.on_observation(
                cursor.observation,
                state
            );
            ++cursor.observation;
        }
        return keep_running;
    }
};

static_assert(std::is_trivially_copyable_v<
    TerminalHybridSchedule::PreparedSchedule
>);
static_assert(std::is_trivially_copyable_v<
    StubbedRegularHybridSchedule::PreparedSchedule
>);
static_assert(std::is_trivially_copyable_v<
    CalendarHybridSchedule<2U>::PreparedSchedule
>);

}  // namespace ai_factory::workbench::volterra
