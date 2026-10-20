// Generated Normal-Inverse-Gaussian device-side parameter law for autonomous model samples.
#pragma once

#include "common/sample/parameter_bounds.cuh"
#ifdef __CUDACC__
#include "common/philox.cuh"
#endif
#include "model/equity/markovian/normal_inverse_gaussian/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

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
                throw std::invalid_argument("Invalid Normal-Inverse-Gaussian sample bounds.");
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
            const float alpha = uniform(configuration.bounds[0], uniforms);
            const float skew_ratio = uniform(configuration.bounds[1], uniforms);
            const float target_volatility = uniform(configuration.bounds[2], uniforms);
            const float beta = skew_ratio * alpha; const float gamma = sqrtf(alpha * alpha - beta * beta); const float delta = target_volatility * target_volatility * gamma * gamma * gamma / (alpha * alpha);
            if (alpha > fmaxf(fabsf(beta + 1.0f), fabsf(beta + 2.0f)) + 0.05f) return {1.0f, 0.0f, 0.0f, alpha, beta, delta};
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

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
