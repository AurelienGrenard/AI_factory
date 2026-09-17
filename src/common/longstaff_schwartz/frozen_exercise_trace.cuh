// Compact central exercise decisions retained for frozen-policy replay.
#pragma once

#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz {

struct FrozenExerciseTrace {
    // Zero-based exercise observation; maturity is regression_count.
    std::uint32_t observation;
    float spot;
};

enum class InitialExerciseDecision : std::uint8_t {
    continuation,
    exercise,
    invalid,
};

}  // namespace ai_factory::workbench::longstaff_schwartz
