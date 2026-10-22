// Generated G2 device-side parameter law for autonomous model samples.
#pragma once

#include "sample/parameter_bounds.cuh"
#ifdef __CUDACC__
#include "philox/distributions.cuh"
#endif
#include "model/fixed_income/g2/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::g2 {

struct SampleParameterSampler {
    using Parameters = ModelParameters;
    struct Configuration {
        sample::ParameterUniformBounds bounds[7];
    };

    static void validate(const Configuration& configuration) {
        for (const auto& bound : configuration.bounds) {
            if (!std::isfinite(bound.minimum)
                || !std::isfinite(bound.maximum)
                || bound.minimum > bound.maximum) {
                throw std::invalid_argument("Invalid G2 sample bounds.");
            }
        }
    }

#ifdef __CUDACC__
    __device__ __forceinline__ static Parameters sample(
        const Configuration& configuration,
        philox::PhiloxKey key,
        std::uint64_t parameter_index
    ) {
        // Counter-local rejection leaves every parameter row independent and
        // makes a replay insensitive to launch geometry or thread scheduling.
        for (std::uint32_t attempt = 0U; attempt < 4096U; ++attempt) {
            const philox::PhiloxKey attempt_key{
                key.k0 + attempt * 0x9E3779B9U,
                key.k1 + attempt * 0xBB67AE85U,
            };
            philox::UniformSequence uniforms(attempt_key, parameter_index);
            const float mean_reversion_x = uniform(configuration.bounds[0], uniforms);
            const float mean_reversion_gap = uniform(configuration.bounds[1], uniforms);
            const float stationary_standard_deviation_x = uniform(configuration.bounds[2], uniforms);
            const float stationary_standard_deviation_y = uniform(configuration.bounds[3], uniforms);
            const float correlation = uniform(configuration.bounds[4], uniforms);
            const float initial_state_x = uniform(configuration.bounds[5], uniforms);
            const float initial_state_y = uniform(configuration.bounds[6], uniforms);
            const float mean_reversion_y = mean_reversion_x + mean_reversion_gap; const float volatility_x = stationary_standard_deviation_x * sqrtf(2.0f * mean_reversion_x); const float volatility_y = stationary_standard_deviation_y * sqrtf(2.0f * mean_reversion_y);
            if (true) return {{mean_reversion_x, volatility_x, mean_reversion_y, volatility_y, correlation}, {initial_state_x, initial_state_y}};
        }
        asm volatile("trap;");
        return {};
    }

private:
    __device__ __forceinline__ static float uniform(
        sample::ParameterUniformBounds bound,
        philox::UniformSequence& uniforms
    ) {
        return fmaf(
            bound.maximum - bound.minimum,
            uniforms.next(),
            bound.minimum
        );
    }
#endif
};

}  // namespace ai_factory::workbench::model::fixed_income::g2
