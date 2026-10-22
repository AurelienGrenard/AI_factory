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

// All stochastic engines use counter=(path_lo,path_hi,group,domain).
// Domain zero is the single continuous stream; source/step domains are
// selected only by models that need independent variable-consumption streams.
__device__ __forceinline__ PhiloxCounter addressed_random_bits(
    PhiloxKey key,
    std::uint64_t path_index,
    std::uint32_t group_index,
    std::uint32_t domain
) {
    return philox4x32_10(
        key,
        {
            static_cast<std::uint32_t>(path_index),
            static_cast<std::uint32_t>(path_index >> 32U),
            group_index,
            domain,
        }
    );
}

// Preserve the compact single-stream API while enforcing the V2 group bound.
__device__ __forceinline__ PhiloxCounter random_bits(
    PhiloxKey key,
    std::uint64_t path_index,
    std::uint64_t local_group_index
) {
    if (local_group_index > 0xffffffffULL) asm volatile("trap;");
    return addressed_random_bits(
        key, path_index, static_cast<std::uint32_t>(local_group_index), 0U
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
    __device__ __forceinline__ UniformSequence() = default;

    // Reuse the row key and begin at local group zero for this path.
    __device__ __forceinline__ UniformSequence(
        PhiloxKey key,
        std::uint64_t path_index
    ) : key_(key),
        path_index_(path_index),
        values_(uniform_quad(key_, path_index_, local_group_index_++)),
        component_index_(0U) {}

    // Lazy reset is used by cooperative kernels: inactive lanes allocate the
    // same context type without evaluating Philox. The first next() still
    // returns component zero of group zero, exactly as the constructor does.
    __device__ __forceinline__ void reset(
        PhiloxKey key,
        std::uint64_t path_index
    ) {
        key_ = key;
        path_index_ = path_index;
        local_group_index_ = 0ULL;
        values_ = {};
        component_index_ = 4U;
    }

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
    PhiloxKey key_{};
    std::uint64_t path_index_ = 0ULL;
    std::uint64_t local_group_index_ = 0ULL;
    RandomQuad values_{};
    std::uint32_t component_index_ = 4U;
};

// Expose one continuous stream of raw 32-bit Philox words.  Discrete sample
// schedules use this stream instead of converting through FP32, so rejection
// can make every integer in a bounded range exactly equiprobable.
class Uint32Sequence {
public:
    __device__ __forceinline__ Uint32Sequence(
        PhiloxKey key,
        std::uint64_t path_index
    ) : key_(key),
        path_index_(path_index),
        values_(random_bits(key_, path_index_, local_group_index_++)) {}

    __device__ __forceinline__ std::uint32_t next() {
        if (component_index_ == 4U) {
            values_ = random_bits(
                key_, path_index_, local_group_index_++
            );
            component_index_ = 0U;
        }
        const std::uint32_t component_index = component_index_++;
        if (component_index == 0U) return values_.v0;
        if (component_index == 1U) return values_.v1;
        if (component_index == 2U) return values_.v2;
        return values_.v3;
    }

private:
    PhiloxKey key_;
    std::uint64_t path_index_;
    std::uint64_t local_group_index_ = 0ULL;
    PhiloxCounter values_;
    std::uint32_t component_index_ = 0U;
};

// Draw exactly uniformly from [0, bound).  The rejected prefix has size
// 2^32 mod bound, leaving a multiple of bound accepted words.
__device__ __forceinline__ std::uint32_t bounded_uint32(
    Uint32Sequence& integers,
    std::uint32_t bound
) {
    const std::uint32_t rejection_threshold = -bound % bound;
    while (true) {
        const std::uint32_t candidate = integers.next();
        if (candidate >= rejection_threshold) return candidate % bound;
    }
}

}  // namespace ai_factory::workbench::philox
