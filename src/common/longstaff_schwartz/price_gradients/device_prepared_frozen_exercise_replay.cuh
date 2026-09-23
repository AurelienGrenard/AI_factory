// Compatibility include for the explicitly named fixed-step replay policy.
#pragma once

#include "common/longstaff_schwartz/price_gradients/device_prepared_fixed_step_frozen_exercise_replay.cuh"

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

template<typename CoupledDynamics, std::size_t NodeCapacity>
using DevicePreparedFrozenExerciseReplay =
    DevicePreparedFixedStepFrozenExerciseReplay<
        CoupledDynamics,
        NodeCapacity
    >;

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
