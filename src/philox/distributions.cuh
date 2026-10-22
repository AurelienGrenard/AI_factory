// Random laws built on Philox uniform streams.
#pragma once

#include "philox/engine.cuh"

namespace ai_factory::workbench::philox {

// Invert a Poisson CDF from one uniform without consuming a variable stream.
// Pass exp(-poisson_mean) prepared outside the hot path as zero_probability.
__device__ __forceinline__ std::uint32_t poisson_from_uniform(
    float uniform,
    float poisson_mean,
    float zero_probability
) {
    std::uint32_t count = 0U;
    float probability = zero_probability;
    float cumulative = probability;
    while (uniform > cumulative) {
        ++count;
        probability *= poisson_mean / static_cast<float>(count);
        const float next_cumulative = cumulative + probability;
        // The remaining tail can fall below one FP32 ulp while the largest
        // representable uniform is still above the rounded CDF. Return the
        // last representable quantile instead of looping on a stagnant sum.
        if (!(next_cumulative > cumulative)) return count;
        cumulative = next_cumulative;
    }
    return count;
}

// Draw Poisson(mean) from one path-local scalar sequence.
// Inversion is efficient for small means; Hoermann's PTRS transformed
// rejection avoids both linear work and exp(-mean) underflow for large means.
// A finite non-negative mean with a uint32-representable tail is a precondition.
template<typename UniformSource>
__device__ __forceinline__ std::uint32_t poisson_from_uniform_sequence(
    UniformSource& uniforms,
    float poisson_mean
) {
    constexpr float inversion_threshold = 10.0f;
    if (poisson_mean == 0.0f) return 0U;
    if (poisson_mean < inversion_threshold) {
        const float uniform = uniforms.next();
        return poisson_from_uniform(
            uniform, poisson_mean, expf(-poisson_mean)
        );
    }

    // PTRS setup from Hoermann (1993), recomputed because CIR intensities
    // depend on the current path state rather than only on prepared rows.
    const float square_root_mean = sqrtf(poisson_mean);
    const float log_mean = logf(poisson_mean);
    const float b = fmaf(2.53f, square_root_mean, 0.931f);
    const float a = fmaf(0.02483f, b, -0.059f);
    const float inverse_alpha = 1.1239f + 1.1328f / (b - 3.4f);
    const float squeeze_threshold = 0.9277f - 3.6224f / (b - 2.0f);

    while (true) {
        const float centered_uniform = uniforms.next() - 0.5f;
        const float acceptance_uniform = uniforms.next();
        const float distance_to_edge =
            0.5f - fabsf(centered_uniform);
        // The smallest open-interval FP32 uniform can round to exactly -0.5
        // after centering. Reject that measure-zero numerical endpoint.
        if (!(distance_to_edge > 0.0f)) continue;

        const float candidate = floorf(
            (
                2.0f * a / distance_to_edge + b
            ) * centered_uniform
                + poisson_mean
                + 0.43f
        );
        if (distance_to_edge >= 0.07f
            && acceptance_uniform <= squeeze_threshold) {
            return static_cast<std::uint32_t>(candidate);
        }
        if (candidate < 0.0f
            || (distance_to_edge < 0.013f
                && acceptance_uniform > distance_to_edge)) {
            continue;
        }

        const float log_acceptance =
            logf(acceptance_uniform)
            + logf(inverse_alpha)
            - logf(
                a / (distance_to_edge * distance_to_edge) + b
            );
        const float log_probability =
            -poisson_mean
            + candidate * log_mean
            - lgammaf(candidate + 1.0f);
        if (log_acceptance <= log_probability) {
            return static_cast<std::uint32_t>(candidate);
        }
    }
}

// Convert two uniforms into two independent standard normals.
__device__ __forceinline__ NormalPair box_muller(
    float angle_uniform,
    float radius_uniform
) {
    const float radius = sqrtf(-2.0f * logf(radius_uniform));
    const float angle = kTwoPi * angle_uniform;
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    return {radius * cosine, radius * sine};
}

// Cache the second Box-Muller result without owning a separate random stream.
struct NormalPairCache {
    NormalPair normals{};
    bool has_second = false;
};

// Consume scalar uniforms in order and return one cached standard normal.
template<typename UniformSource>
__device__ __forceinline__ float next_normal(
    UniformSource& uniforms,
    NormalPairCache& cache
) {
    if (cache.has_second) {
        cache.has_second = false;
        return cache.normals.second;
    }
    const float angle_uniform = uniforms.next();
    const float radius_uniform = uniforms.next();
    cache.normals = box_muller(angle_uniform, radius_uniform);
    cache.has_second = true;
    return cache.normals.first;
}

// Path-local context for laws consuming uniforms and cached Box-Muller
// normals. The normal cache never owns or restarts a random sequence.
struct NormalRandomContext {
    UniformSequence uniforms;
    NormalPairCache normals;

    __device__ __forceinline__ NormalRandomContext() = default;

    __device__ __forceinline__ NormalRandomContext(
        PhiloxKey key,
        std::uint64_t path_index
    ) : uniforms(key, path_index) {}

    __device__ __forceinline__ void reset(
        PhiloxKey key,
        std::uint64_t path_index
    ) {
        uniforms.reset(key, path_index);
        normals = {};
    }
};

namespace detail {

// Marsaglia-Tsang core for a unit-scale Gamma shape greater than or equal to 1.
template<typename UniformSource>
__device__ __forceinline__ float marsaglia_tsang_gamma_shape_at_least_one(
    UniformSource& uniforms,
    NormalPairCache& normal_cache,
    float shape
) {
    const float d = shape - 1.0f / 3.0f;
    const float c = 1.0f / sqrtf(9.0f * d);

    while (true) {
        const float normal = next_normal(uniforms, normal_cache);
        const float one_plus_cx = fmaf(c, normal, 1.0f);
        if (one_plus_cx <= 0.0f) continue;

        const float candidate = one_plus_cx * one_plus_cx * one_plus_cx;
        const float uniform = uniforms.next();
        const float normal2 = normal * normal;
        const float normal4 = normal2 * normal2;
        if (uniform < 1.0f - 0.0331f * normal4) {
            return d * candidate;
        }
        if (logf(uniform)
            < 0.5f * normal2
                + d * (1.0f - candidate + logf(candidate))) {
            return d * candidate;
        }
    }
}

}  // namespace detail

// Draw Gamma(shape, scale) with the Marsaglia-Tsang rejection method.
// Positive shape and scale are preconditions validated by the caller.
template<typename UniformSource>
__device__ __forceinline__ float marsaglia_tsang_gamma(
    UniformSource& uniforms,
    NormalPairCache& normal_cache,
    float shape,
    float scale
) {
    if (shape >= 1.0f) {
        return scale * detail::marsaglia_tsang_gamma_shape_at_least_one(
            uniforms, normal_cache, shape
        );
    }

    // Boost a sub-unit shape, then apply the exact power transformation.
    const float boost_uniform = uniforms.next();
    const float boosted_gamma =
        detail::marsaglia_tsang_gamma_shape_at_least_one(
            uniforms, normal_cache, shape + 1.0f
        );
    return scale * boosted_gamma
        * expf(logf(boost_uniform) / shape);
}

// Draw scale * X for X following a non-central chi-square distribution.
// The Poisson-Gamma mixture is exact:
//   N ~ Poisson(noncentrality / 2)
//   scale * X ~ Gamma(degrees_of_freedom / 2 + N, 2 * scale).
// Positive degrees_of_freedom and scale, and finite non-negative
// noncentrality, are preconditions validated by the caller.
template<typename UniformSource>
__device__ __forceinline__ float scaled_noncentral_chi_square(
    UniformSource& uniforms,
    NormalPairCache& normal_cache,
    float degrees_of_freedom,
    float noncentrality,
    float scale
) {
    const std::uint32_t poisson = poisson_from_uniform_sequence(
        uniforms,
        0.5f * noncentrality
    );
    return marsaglia_tsang_gamma(
        uniforms,
        normal_cache,
        0.5f * degrees_of_freedom + static_cast<float>(poisson),
        2.0f * scale
    );
}

// Transform the two Michael-Schucany-Haas primitives into IG(mean, shape).
// Keeping this pure transformation public lets coupled simulations replay the
// same fixed-cost primitives under several valid parameter sets.
__device__ __forceinline__ float
michael_schucany_haas_inverse_gaussian_from_variates(
    float normal,
    float selection_uniform,
    float mean,
    float shape
) {
    const float squared_normal = normal * normal;
    const float w = mean * squared_normal / (2.0f * shape);
    const float root = sqrtf(w * (2.0f + w));
    const float ratio = 1.0f + w + root;
    const float lower_candidate = mean / ratio;
    const float lower_probability = 1.0f - 1.0f / (ratio + 1.0f);
    return selection_uniform <= lower_probability
        ? lower_candidate
        : mean * ratio;
}

// Draw IG(mean, shape) with the Michael-Schucany-Haas exact construction.
// The reciprocal-root form avoids cancellation in the smaller candidate.
template<typename UniformSource>
__device__ __forceinline__ float michael_schucany_haas_inverse_gaussian(
    UniformSource& uniforms,
    NormalPairCache& normal_cache,
    float mean,
    float shape
) {
    const float normal = next_normal(uniforms, normal_cache);
    const float selection_uniform = uniforms.next();
    return michael_schucany_haas_inverse_gaussian_from_variates(
        normal, selection_uniform, mean, shape
    );
}

}  // namespace ai_factory::workbench::philox
