// Relative full-width spot bumps and paired payoff results; no RNG ownership.
#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::equity::price_delta {

struct SpotBumpConfiguration {
    float relative_width = 0.01f;  // S0 +/- relative_width * S0 / 2.
};

inline void validate_device_context(SpotBumpConfiguration configuration) {
    if (!std::isfinite(configuration.relative_width)
        || !(configuration.relative_width > 0.0f)
        || !(configuration.relative_width < 2.0f)) {
        throw std::invalid_argument("Spot bump width must be finite and in (0, 2).");
    }
}

struct SpotBump {
    float central;
    float lower;
    float upper;
    float width;
};

__host__ __device__ inline SpotBump prepare_spot_bump(
    float spot, SpotBumpConfiguration configuration
) {
    const float half_width = (0.5f * configuration.relative_width) * spot;
    // Explicit rounded operations keep host/device endpoints identical.
    #ifdef __CUDA_ARCH__
    const float lower = __fsub_rn(spot, half_width);
    const float upper = __fadd_rn(spot, half_width);
    #else
    const float lower = spot - half_width;
    const float upper = spot + half_width;
    #endif
    return {spot, lower, upper, upper - lower};
}

inline void validate_spot_bump(float spot, SpotBumpConfiguration configuration) {
    validate_device_context(configuration);
    const auto bump = prepare_spot_bump(spot, configuration);
    if (!std::isfinite(spot) || !std::isfinite(bump.upper)
        || !std::isfinite(bump.width) || !(bump.lower > 0.0f)
        || !(bump.lower < spot) || !(spot < bump.upper)) {
        throw std::invalid_argument("Spot bump requires distinct positive finite endpoints.");
    }
}

struct PairedPayoff {
    float price;
    float delta;
};

// Preserve native log-spot observations without an exp/log round trip.
struct SpotObservation {
    float spot;
    float log_spot;
};

struct SpotObservationPolicy {
    using State = SpotObservation;
    static constexpr bool kNativeLogSpot = true;
    __device__ __forceinline__ static float spot(const State& state) {
        return state.spot;
    }
    __device__ __forceinline__ static float log_spot(const State& state) {
        return state.log_spot;
    }
};

}  // namespace ai_factory::workbench::equity::price_delta
