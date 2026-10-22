// Generated CIR device-side parameter law for autonomous model samples.
#pragma once

#include "sample/parameter_bounds.cuh"
#ifdef __CUDACC__
#include "philox/distributions.cuh"
#endif
#include "model/fixed_income/cir/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::cir {

struct SampleParameterSampler {
    using Parameters = ModelParameters;
    struct Configuration {
        sample::ParameterUniformBounds bounds[3];
    };

    static void validate(const Configuration& configuration) {
        for (const auto& bound : configuration.bounds) {
            if (!std::isfinite(bound.minimum)
                || !std::isfinite(bound.maximum)
                || bound.minimum > bound.maximum) {
                throw std::invalid_argument("Invalid CIR sample bounds.");
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
            const float mean_reversion = uniform(configuration.bounds[0], uniforms);
            const float long_term_mean = uniform(configuration.bounds[1], uniforms);
            const float initial_state = uniform(configuration.bounds[2], uniforms);
            const float volatility = uniform({fmaxf(sqrtf(mean_reversion * long_term_mean / 5.0f), 0.005f), fminf(sqrtf(12.0f * mean_reversion * long_term_mean), 0.30f)}, uniforms);
            if (true) return {{mean_reversion, long_term_mean, volatility}, initial_state};
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

}  // namespace ai_factory::workbench::model::fixed_income::cir
