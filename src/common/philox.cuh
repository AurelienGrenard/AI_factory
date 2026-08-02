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
    PhiloxCounter counter,
    PhiloxKey key
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

// Build one counter group from a pre-split key and group index.
__device__ __forceinline__ PhiloxCounter random_bits(
    PhiloxKey key,
    std::uint64_t group_index
) {
    return philox4x32_10(
        {
            static_cast<std::uint32_t>(group_index),
            static_cast<std::uint32_t>(group_index >> 32U),
            0U,
            0U,
        },
        key
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
    std::uint64_t group_index
) {
    const PhiloxCounter bits = random_bits(key, group_index);
    return {
        uint32_to_uniform(bits.v0),
        uint32_to_uniform(bits.v1),
        uint32_to_uniform(bits.v2),
        uint32_to_uniform(bits.v3),
    };
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

// Cache four uniforms at a time for one path aligned on a Philox group.
class UniformSequence {
public:
    // Reuse the row key and begin at the path's first complete group.
    __device__ __forceinline__ UniformSequence(
        PhiloxKey key,
        std::uint64_t first_group
    ) : key_(key),
        group_index_(first_group),
        values_(uniform_quad(key, first_group)) {}

    // Return the next uniform, refreshing the cached group when necessary.
    __device__ __forceinline__ float next() {
        if (component_index_ == 4U) {
            ++group_index_;
            values_ = uniform_quad(key_, group_index_);
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
    std::uint64_t group_index_;
    RandomQuad values_;
    std::uint32_t component_index_ = 0U;
};

// Convert the uniform stream into cached pairs of standard normals.
class NormalSequence {
public:
    // Begin at the first complete Philox group reserved for one path.
    __device__ __forceinline__ NormalSequence(
        PhiloxKey key,
        std::uint64_t first_group
    ) : uniforms_(key, first_group) {}

    // Return one normal and reuse the second value from each Box-Muller pair.
    __device__ __forceinline__ float next() {
        if (has_second_) {
            has_second_ = false;
            return normals_.second;
        }
        const float angle_uniform = uniforms_.next();
        const float radius_uniform = uniforms_.next();
        normals_ = box_muller(angle_uniform, radius_uniform);
        has_second_ = true;
        return normals_.first;
    }

private:
    UniformSequence uniforms_;
    NormalPair normals_{};
    bool has_second_ = false;
};

}  // namespace ai_factory::workbench::philox
