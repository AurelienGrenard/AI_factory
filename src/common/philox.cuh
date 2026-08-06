// Counter-based Philox utilities generating FP32 uniforms by groups of four.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::philox {

// Constants used by the ten Philox rounds and the FP32 conversion.
constexpr std::uint32_t kM0 = 0xD2511F53U;
constexpr std::uint32_t kM1 = 0xCD9E8D57U;
constexpr std::uint32_t kW0 = 0x9E3779B9U;
constexpr std::uint32_t kW1 = 0xBB67AE85U;
constexpr float kUInt32Scale = 0x1p-32f;
constexpr float kHalfUInt32Scale = 0x1p-33f;
constexpr float kLargestUniform = 0x1.fffffep-1f;
constexpr float kTwoPi = 6.2831853071795864769f;

// Four 32-bit words form the Philox counter transformed by each round.
struct PhiloxCounter {
    std::uint32_t v0;
    std::uint32_t v1;
    std::uint32_t v2;
    std::uint32_t v3;
};

// Two 32-bit words form the Philox key derived from the 64-bit seed.
struct PhiloxKey {
    std::uint32_t k0;
    std::uint32_t k1;
};

// Split one public 64-bit seed once before repeated Philox invocations.
__device__ __forceinline__ PhiloxKey make_key(std::uint64_t seed) {
    return {
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32U),
    };
}

// One Philox invocation is converted into four FP32 random values at once.
struct RandomQuad {
    float first;
    float second;
    float third;
    float fourth;
};

// Hold the two independent normals returned by Box-Muller.
struct NormalPair {
    float first;
    float second;
};

// Apply the ten integer-only mixing rounds of Philox-4x32-10.
__device__ __forceinline__ PhiloxCounter philox4x32_10(
    PhiloxKey key,
    PhiloxCounter counter
) {
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        // Read both halves of each 32-bit integer product.
        const std::uint32_t hi0 = __umulhi(kM0, counter.v0);
        const std::uint32_t hi1 = __umulhi(kM1, counter.v2);
        const std::uint32_t lo0 = kM0 * counter.v0;
        const std::uint32_t lo1 = kM1 * counter.v2;
        counter = {
            static_cast<std::uint32_t>(hi1 ^ counter.v1 ^ key.k0),
            lo1,
            static_cast<std::uint32_t>(hi0 ^ counter.v3 ^ key.k1),
            lo0,
        };
        if (round != 9) {
            key.k0 += kW0;
            key.k1 += kW1;
        }
    }
    return counter;
}

// Address one random group by an independent path and its local group index.
__device__ __forceinline__ PhiloxCounter random_bits(
    PhiloxKey key,
    std::uint64_t path_index,
    std::uint64_t local_group_index
) {
    return philox4x32_10(
        key,
        {
            static_cast<std::uint32_t>(path_index),
            static_cast<std::uint32_t>(path_index >> 32U),
            static_cast<std::uint32_t>(local_group_index),
            static_cast<std::uint32_t>(local_group_index >> 32U),
        }
    );
}

// Convert one unsigned 32-bit word into a uniform value strictly inside (0, 1).
__device__ __forceinline__ float uint32_to_uniform(std::uint32_t value) {
    // Midpoint scaling avoids zero; the clamp avoids FP32 rounding to one.
    const float uniform =
        fmaf(__uint2float_rn(value), kUInt32Scale, kHalfUInt32Scale);
    return fminf(uniform, kLargestUniform);
}

// Generate four uniforms without rebuilding the Philox key.
__device__ __forceinline__ RandomQuad uniform_quad(
    PhiloxKey key,
    std::uint64_t path_index,
    std::uint64_t local_group_index
) {
    const PhiloxCounter bits = random_bits(
        key, path_index, local_group_index
    );
    return {
        uint32_to_uniform(bits.v0),
        uint32_to_uniform(bits.v1),
        uint32_to_uniform(bits.v2),
        uint32_to_uniform(bits.v3),
    };
}

// Expose one continuous scalar stream while keeping Philox groups internal.
class UniformSequence {
public:
    // Reuse the row key and begin at local group zero for this path.
    __device__ __forceinline__ UniformSequence(
        PhiloxKey key,
        std::uint64_t path_index
    ) : key_(key),
        path_index_(path_index),
        values_(uniform_quad(key_, path_index_, local_group_index_++)) {}

    // Return the next uniform, refreshing the cached group when necessary.
    __device__ __forceinline__ float next() {
        if (component_index_ == 4U) {
            values_ = uniform_quad(
                key_, path_index_, local_group_index_++
            );
            component_index_ = 0U;
        }
        const std::uint32_t component_index = component_index_++;
        if (component_index == 0U) return values_.first;
        if (component_index == 1U) return values_.second;
        if (component_index == 2U) return values_.third;
        return values_.fourth;
    }

private:
    PhiloxKey key_;
    std::uint64_t path_index_;
    std::uint64_t local_group_index_ = 0ULL;
    RandomQuad values_;
    std::uint32_t component_index_ = 0U;
};

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
__device__ __forceinline__ float next_normal(
    UniformSequence& uniforms,
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

namespace detail {

// Marsaglia-Tsang core for a unit-scale Gamma shape greater than or equal to 1.
__device__ __forceinline__ float marsaglia_tsang_gamma_shape_at_least_one(
    UniformSequence& uniforms,
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
__device__ __forceinline__ float marsaglia_tsang_gamma(
    UniformSequence& uniforms,
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

// Draw IG(mean, shape) with the Michael-Schucany-Haas exact construction.
// The reciprocal-root form avoids cancellation in the smaller candidate.
__device__ __forceinline__ float michael_schucany_haas_inverse_gaussian(
    UniformSequence& uniforms,
    NormalPairCache& normal_cache,
    float mean,
    float shape
) {
    const float normal = next_normal(uniforms, normal_cache);
    const float selection_uniform = uniforms.next();
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

}  // namespace ai_factory::workbench::philox
