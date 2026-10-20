// Generated Rough-Heston device-side parameter law for autonomous model samples.
#pragma once

#include "common/sample/parameter_bounds.cuh"
#ifdef __CUDACC__
#include "common/philox.cuh"
#endif
#include "model/equity/rough/rough_heston/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::rough_heston {

struct SampleParameterSampler {
    using Parameters = ModelParameters;
    struct Configuration {
        sample::ParameterUniformBounds bounds[6];
    };

    static void validate(const Configuration& configuration) {
        for (const auto& bound : configuration.bounds) {
            if (!std::isfinite(bound.minimum)
                || !std::isfinite(bound.maximum)
                || bound.minimum > bound.maximum) {
                throw std::invalid_argument("Invalid Rough-Heston sample bounds.");
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
            const float initial_variance = uniform(configuration.bounds[1], uniforms);
            const float mean_reversion = uniform(configuration.bounds[2], uniforms);
            const float long_run_variance = uniform(configuration.bounds[3], uniforms);
            const float hurst_exponent = uniform(configuration.bounds[4], uniforms);
            const float rho = uniform(configuration.bounds[5], uniforms);
            const float variance_drift = mean_reversion * long_run_variance; const float volatility_of_variance = uniform({fmaxf(sqrtf(variance_drift / 5.0f), 0.08f), fminf(sqrtf(12.0f * variance_drift), 0.8f)}, uniforms);
            if (true) return {spot, 0.0f, 0.0f, initial_variance, mean_reversion, variance_drift, volatility_of_variance, hurst_exponent, rho};
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

}  // namespace ai_factory::workbench::model::equity::rough_heston
