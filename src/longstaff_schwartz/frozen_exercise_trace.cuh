// Compact central exercise decisions retained for frozen-policy replay.
#pragma once

#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz {

struct FrozenExerciseTrace {
    // Zero-based exercise observation; maturity is regression_count.
    std::uint32_t observation;
    float spot;
};

// Product-independent stopping time retained by a central LSM solve. A replay
// reconstructs every bumped state from its Philox address, so fixed-income
// products only need the selected contractual observation.
struct FrozenExerciseIndex {
    std::uint32_t observation;
};

enum class InitialExerciseDecision : std::uint8_t {
    continuation,
    exercise,
    invalid,
};

}  // namespace ai_factory::workbench::longstaff_schwartz
