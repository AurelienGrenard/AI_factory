// Host-visible choice of stopping-rule replay for early-exercise sensitivities.
#pragma once

#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

enum class ExerciseReplayStrategy : std::uint8_t {
    frozen_exercise_time,
    frozen_regression_policy,
};

constexpr std::string_view to_string(ExerciseReplayStrategy strategy) {
    switch (strategy) {
    case ExerciseReplayStrategy::frozen_exercise_time:
        return "frozen_exercise_time";
    case ExerciseReplayStrategy::frozen_regression_policy:
        return "frozen_regression_policy";
    }
    return "unknown";
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
