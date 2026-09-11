// Host-side boundary tests for calendar and fixed-step launch validation.
#include "common/simulation/early_exercise_schedule.cuh"
#include "common/simulation/schedule.cuh"
#include "common/sample/sources.cuh"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace simulation = ai_factory::workbench::simulation;
namespace sample = ai_factory::workbench::sample;

namespace {

template<typename Exception, typename Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

}  // namespace

int main() {
    const simulation::FixedStepTimeConfiguration boundary{
        1.0f / 252.0f,
        1U << 31U,
    };

    if (simulation::checked_fixed_step_transition_count(1U, boundary)
            != (1U << 31U)
        || !throws<std::overflow_error>([&] {
            simulation::checked_fixed_step_transition_count(2U, boundary);
        })) {
        std::cerr << "fixed-step checked multiplication boundary failed\n";
        return 1;
    }

    simulation::validate_calendar(simulation::MaturityCalendar{1U}, boundary);
    simulation::validate_calendar(
        simulation::RegularCalendar{1U, 2U},
        boundary
    );
    simulation::validate_calendar(
        simulation::StubbedRegularCalendar{1U, 1U, 2U},
        boundary
    );
    simulation::validate_calendar(
        simulation::StaticCalendar<2U>{{1U, 1U}},
        boundary
    );
    simulation::validate_exercise_calendar(
        simulation::RegularExerciseCalendar{1U, 1U, 2U},
        boundary
    );

    if (!throws<std::overflow_error>([&] {
            simulation::validate_calendar(
                simulation::MaturityCalendar{2U},
                boundary
            );
        })
        || !throws<std::overflow_error>([&] {
            simulation::validate_calendar(
                simulation::RegularCalendar{2U, 2U},
                boundary
            );
        })
        || !throws<std::overflow_error>([&] {
            simulation::validate_calendar(
                simulation::StubbedRegularCalendar{2U, 1U, 2U},
                boundary
            );
        })
        || !throws<std::overflow_error>([&] {
            simulation::validate_calendar(
                simulation::StaticCalendar<2U>{{1U, 2U}},
                boundary
            );
        })
        || !throws<std::overflow_error>([&] {
            simulation::validate_exercise_calendar(
                simulation::RegularExerciseCalendar{2U, 1U, 2U},
                boundary
            );
        })
        || !throws<std::overflow_error>([&] {
            simulation::validate_exercise_calendar(
                simulation::MaturityAlignedExerciseCalendar{2U, 1U},
                boundary
            );
        })
        || !throws<std::overflow_error>([&] {
            sample::UniformMaturityCalendarSource{{1U, 2U}, nullptr}
                .validate(1U, boundary);
        })
        || !throws<std::overflow_error>([&] {
            sample::RandomIncreasingCalendarSource<
                simulation::StaticCalendar<3U>,
                3U
            >{{1U, 3U, 1U}, nullptr, 0U}.validate(1U, boundary);
        })) {
        std::cerr << "a calendar overflow was not rejected on the host\n";
        return 1;
    }

    const simulation::FixedStepTimeConfiguration non_finite_fixed{
        std::numeric_limits<float>::max(),
        2U,
    };
    if (!throws<std::overflow_error>([&] {
            simulation::validate_calendar(
                simulation::MaturityCalendar{1U},
                non_finite_fixed
            );
        })) {
        std::cerr << "non-finite fixed-step year fraction was accepted\n";
        return 1;
    }

    const simulation::ExactTransitionTimeConfiguration non_finite_exact{
        std::numeric_limits<float>::max(),
    };
    if (!throws<std::overflow_error>([&] {
            simulation::validate_calendar(
                simulation::MaturityCalendar{
                    std::numeric_limits<std::uint32_t>::max()
                },
                non_finite_exact
            );
        })) {
        std::cerr << "non-finite exact-transition year fraction was accepted\n";
        return 1;
    }

    return 0;
}
