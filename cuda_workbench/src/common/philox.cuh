// Counter-based Philox random-number utilities used inside CUDA kernels.
// This file maps deterministic integer counters to FP32 uniform and normal
// samples, then exposes small cached sequences for path simulation.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::rng {

// Philox-4x32-10 is counter based: a random value is determined by a seed,
// stream, and integer index. Thread scheduling and block size therefore do not
// change the random experiment. These constants define its rounds and the
// integer-to-FP32 conversion.
constexpr std::uint32_t kM0 = 0xD2511F53U;
constexpr std::uint32_t kM1 = 0xCD9E8D57U;
constexpr std::uint32_t kW0 = 0x9E3779B9U;
constexpr std::uint32_t kW1 = 0xBB67AE85U;
constexpr float kUInt32Scale = 0x1p-32f;
constexpr float kHalfUInt32Scale = 0x1p-33f;
constexpr float kLargestUniform = 0x1.fffffep-1f;
constexpr float kTwoPi = 6.2831853071795864769f;

// Four 32-bit words form the Philox counter transformed by each round.
struct Counter {
    std::uint32_t v0;
    std::uint32_t v1;
    std::uint32_t v2;
    std::uint32_t v3;
};

// Two 32-bit words form the Philox key derived from the 64-bit seed.
struct Key {
    std::uint32_t k0;
    std::uint32_t k1;
};

// One Philox invocation is converted into four FP32 random values at once.
struct RandomQuad {
    float first;
    float second;
    float third;
    float fourth;
};

// Apply the ten integer-only mixing rounds of Philox-4x32-10.
__device__ __forceinline__ Counter philox4x32_10(Counter counter, Key key) {
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        // __umulhi returns the high 32 bits while normal multiplication keeps
        // the low 32 bits. No floating-point arithmetic is involved here.
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

// Split seed, stream, and counter index into the words consumed by Philox.
__device__ __forceinline__ Counter random_bits(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t block_index
) {
    return philox4x32_10(
        {
            static_cast<std::uint32_t>(block_index),
            static_cast<std::uint32_t>(block_index >> 32U),
            static_cast<std::uint32_t>(stream),
            static_cast<std::uint32_t>(stream >> 32U),
        },
        {
            static_cast<std::uint32_t>(seed),
            static_cast<std::uint32_t>(seed >> 32U),
        }
    );
}

// Convert one unsigned 32-bit word into a uniform value strictly inside (0, 1).
__device__ __forceinline__ float uint32_to_uniform(std::uint32_t value) {
    // The midpoint convention avoids zero. FP32 rounding can map the largest
    // uint32 to one, so clamp to the greatest representable float below one.
    const float uniform =
        fmaf(__uint2float_rn(value), kUInt32Scale, kHalfUInt32Scale);
    return fminf(uniform, kLargestUniform);
}

// Generate four independent uniforms from one counter position.
__device__ __forceinline__ RandomQuad uniform_quad(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t block_index
) {
    const Counter bits = random_bits(seed, stream, block_index);
    return {
        uint32_to_uniform(bits.v0),
        uint32_to_uniform(bits.v1),
        uint32_to_uniform(bits.v2),
        uint32_to_uniform(bits.v3),
    };
}

// Transform four uniforms into four standard normals with Box-Muller.
__device__ __forceinline__ RandomQuad normal_quad(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t block_index
) {
    const RandomQuad uniforms = uniform_quad(seed, stream, block_index);
    const float radius0 = sqrtf(-2.0f * logf(uniforms.first));
    const float angle0 = kTwoPi * uniforms.second;
    const float radius1 = sqrtf(-2.0f * logf(uniforms.third));
    const float angle1 = kTwoPi * uniforms.fourth;

    float sine0 = 0.0f;
    float cosine0 = 0.0f;
    float sine1 = 0.0f;
    float cosine1 = 0.0f;
    sincosf(angle0, &sine0, &cosine0);
    sincosf(angle1, &sine1, &cosine1);
    return {
        radius0 * cosine0,
        radius0 * sine0,
        radius1 * cosine1,
        radius1 * sine1,
    };
}

// A sequence caches four transformed values. The first index is normally
// path * num_steps, so every path owns a deterministic, non-overlapping range.
class NormalSequence {
public:
    // Start a deterministic normal stream at the requested scalar index.
    __device__ __forceinline__ NormalSequence(
        std::uint64_t seed,
        std::uint64_t stream,
        std::uint64_t first_index
    ) : seed_(seed), stream_(stream), index_(first_index) {}

    // Return the next normal, refreshing the cached group every four values.
    __device__ __forceinline__ float next() {
        const std::uint64_t block_index = index_ / 4ULL;
        if (!has_block_ || block_index != cached_block_index_) {
            values_ = normal_quad(seed_, stream_, block_index);
            cached_block_index_ = block_index;
            has_block_ = true;
        }
        return component(values_, index_++ % 4ULL);
    }

private:
    // Select one scalar from a cached group of four values.
    __device__ __forceinline__ static float component(
        const RandomQuad& values,
        std::uint64_t index
    ) {
        if (index == 0ULL) return values.first;
        if (index == 1ULL) return values.second;
        if (index == 2ULL) return values.third;
        return values.fourth;
    }

    std::uint64_t seed_;
    std::uint64_t stream_;
    std::uint64_t index_;
    std::uint64_t cached_block_index_ = 0ULL;
    RandomQuad values_{};
    bool has_block_ = false;
};

// UniformSequence provides the same cached sequential interface for uniforms.
class UniformSequence {
public:
    // Start a deterministic uniform stream at the requested scalar index.
    __device__ __forceinline__ UniformSequence(
        std::uint64_t seed,
        std::uint64_t stream,
        std::uint64_t first_index
    ) : seed_(seed), stream_(stream), index_(first_index) {}

    // Return the next uniform, refreshing the cached group when necessary.
    __device__ __forceinline__ float next() {
        const std::uint64_t block_index = index_ / 4ULL;
        if (!has_block_ || block_index != cached_block_index_) {
            values_ = uniform_quad(seed_, stream_, block_index);
            cached_block_index_ = block_index;
            has_block_ = true;
        }
        const std::uint64_t component_index = index_++ % 4ULL;
        if (component_index == 0ULL) return values_.first;
        if (component_index == 1ULL) return values_.second;
        if (component_index == 2ULL) return values_.third;
        return values_.fourth;
    }

private:
    std::uint64_t seed_;
    std::uint64_t stream_;
    std::uint64_t index_;
    std::uint64_t cached_block_index_ = 0ULL;
    RandomQuad values_{};
    bool has_block_ = false;
};

}  // namespace ai_factory::workbench::rng
