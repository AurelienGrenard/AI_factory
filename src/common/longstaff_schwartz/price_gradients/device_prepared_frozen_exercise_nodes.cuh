// Shared bounded-node storage for fixed and exact frozen-exercise replays.
#pragma once

#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "common/price_gradients/sensitivity_task.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

template<std::size_t NodeCapacity>
struct FrozenExerciseNodeMetadata {
    static_assert(NodeCapacity == 3U || NodeCapacity == 4U);
    static constexpr std::size_t kEndpointCapacity = NodeCapacity - 1U;

    float spot_scales[kEndpointCapacity]{};
    bool reuse_central[kEndpointCapacity]{};
    std::uint8_t node_count = 0U;
    std::uint8_t endpoint_count = 0U;
};

template<typename Scenario, std::size_t NodeCapacity>
__device__ __forceinline__ FrozenExerciseNodeMetadata<NodeCapacity>
prepare_frozen_exercise_node_metadata(
    const ::ai_factory::workbench::price_gradients::SensitivityNodes<
        Scenario,
        NodeCapacity
    >& nodes,
    std::size_t node_count
) {
    FrozenExerciseNodeMetadata<NodeCapacity> result{};
    result.node_count = static_cast<std::uint8_t>(node_count);
    result.endpoint_count = static_cast<std::uint8_t>(node_count - 1U);
    #pragma unroll
    for (std::size_t endpoint = 0U;
         endpoint < result.kEndpointCapacity;
         ++endpoint) {
        const std::size_t node = endpoint + 1U;
        if (node >= node_count) continue;
        result.spot_scales[endpoint] = nodes[node].spot_scale;
        result.reuse_central[endpoint] = nodes[node].reuse_central;
    }
    return result;
}

template<typename Dynamics, std::size_t NodeCapacity>
__device__ __forceinline__ bool initialize_frozen_exercise_replay(
    const FrozenExerciseNodeMetadata<NodeCapacity>& metadata,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    longstaff_schwartz::FrozenExerciseTrace exercise,
    ::ai_factory::workbench::price_gradients::SensitivityValues<NodeCapacity>&
        spots,
    typename Dynamics::State (&states)[NodeCapacity - 1U]
) {
    spots[0U] = exercise.spot;
    bool requires_replay = false;
    #pragma unroll
    for (std::size_t endpoint = 0U;
         endpoint < NodeCapacity - 1U;
         ++endpoint) {
        const std::size_t node = endpoint + 1U;
        if (endpoint >= metadata.endpoint_count) continue;
        if (metadata.reuse_central[endpoint]) {
            spots[node] = exercise.spot * metadata.spot_scales[endpoint];
        } else {
            states[endpoint] = Dynamics::initial(prepared[node]);
            requires_replay = true;
        }
    }
    return requires_replay;
}

template<typename Dynamics, std::size_t NodeCapacity>
__device__ __forceinline__ void finish_frozen_exercise_replay(
    const FrozenExerciseNodeMetadata<NodeCapacity>& metadata,
    const typename Dynamics::State (&states)[NodeCapacity - 1U],
    ::ai_factory::workbench::price_gradients::SensitivityValues<NodeCapacity>&
        spots
) {
    #pragma unroll
    for (std::size_t endpoint = 0U;
         endpoint < NodeCapacity - 1U;
         ++endpoint) {
        const std::size_t node = endpoint + 1U;
        if (endpoint < metadata.endpoint_count
            && !metadata.reuse_central[endpoint]) {
            spots[node] = Dynamics::spot(states[endpoint])
                * metadata.spot_scales[endpoint];
        }
    }
}

struct FrozenExerciseReplayTime {
    float numerical_step;
    float first_exercise_time;
    float exercise_interval;
};

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
