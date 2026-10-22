// Compile-time barrier predicates for any scalar observable.
#pragma once

#include <cstdint>

namespace ai_factory::workbench::payoff {

enum class BarrierDirection : std::uint8_t {
    down,
    up,
};

template<BarrierDirection Direction>
__device__ __forceinline__ bool barrier_breached(
    float value,
    float barrier
) {
    if constexpr (Direction == BarrierDirection::down) {
        return value <= barrier;
    } else {
        return value >= barrier;
    }
}

}  // namespace ai_factory::workbench::payoff
