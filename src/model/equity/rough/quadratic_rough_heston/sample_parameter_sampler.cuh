// Generated Quadratic rough-Heston device-side parameter law for autonomous model samples.
#pragma once

#include "sample/parameter_bounds.cuh"
#ifdef __CUDACC__
#include "philox/distributions.cuh"
#endif
#include "model/equity/rough/quadratic_rough_heston/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

struct SampleParameterSampler {
    using Parameters = ModelParameters;
    struct Configuration {
        sample::ParameterUniformBounds bounds[8];
    };

    static void validate(const Configuration& configuration) {
        for (const auto& bound : configuration.bounds) {
            if (!std::isfinite(bound.minimum)
                || !std::isfinite(bound.maximum)
                || bound.minimum > bound.maximum) {
                throw std::invalid_argument("Invalid Quadratic rough-Heston sample bounds.");
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
            const float spot = uniform(configuration.bounds[0], uniforms);
            const float initial_feedback = uniform(configuration.bounds[1], uniforms);
            const float quadratic_scale = uniform(configuration.bounds[2], uniforms);
            const float quadratic_shift = uniform(configuration.bounds[3], uniforms);
            const float variance_floor = uniform(configuration.bounds[4], uniforms);
            const float feedback_rate = uniform(configuration.bounds[5], uniforms);
            const float feedback_volatility = uniform(configuration.bounds[6], uniforms);
            const float hurst_exponent = uniform(configuration.bounds[7], uniforms);

            if (true) return {spot, 0.0f, 0.0f, initial_feedback, quadratic_scale, quadratic_shift, variance_floor, feedback_rate, feedback_volatility, hurst_exponent};
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

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
