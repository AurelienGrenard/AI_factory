// Optional source/step domains over the common Philox V2 counter layout.
#pragma once

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::philox {

// The zero domain is used by the single continuous stream in philox.cuh.
// Positive source identifiers
// occupy the high byte and the 24-bit step index occupies the low bits.
inline constexpr std::uint32_t kMaximumDomainStep = 0x00ffffffU;

template<std::uint8_t SourceId>
__device__ __forceinline__ std::uint32_t source_step_domain(
    std::uint32_t step_index
) {
    static_assert(SourceId != 0U, "source zero is reserved for the fixed stream");
    if (step_index > kMaximumDomainStep) asm volatile("trap;");
    return (static_cast<std::uint32_t>(SourceId) << 24U) | step_index;
}

__device__ __forceinline__ PhiloxCounter domain_random_bits(
    PhiloxKey key,
    std::uint64_t path_index,
    std::uint32_t group_index,
    std::uint32_t domain
) {
    return addressed_random_bits(key, path_index, group_index, domain);
}

__device__ __forceinline__ RandomQuad domain_uniform_quad(
    PhiloxKey key,
    std::uint64_t path_index,
    std::uint32_t group_index,
    std::uint32_t domain
) {
    const auto bits = domain_random_bits(key, path_index, group_index, domain);
    return {
        uint32_to_uniform(bits.v0), uint32_to_uniform(bits.v1),
        uint32_to_uniform(bits.v2), uint32_to_uniform(bits.v3),
    };
}

// Construct this small stream only while its source is used. Rejections in
// this source/step cannot shift another source or the next step.
class DomainUniformSequence {
public:
    __device__ __forceinline__ DomainUniformSequence(
        PhiloxKey key,
        std::uint64_t path_index,
        std::uint32_t domain
    ) : key_(key), path_index_(path_index), domain_(domain) {}

    __device__ __forceinline__ float next() {
        if (component_index_ == 4U) {
            if (next_group_ > 0xffffffffULL) {
                asm volatile("trap;");
            }
            values_ = domain_uniform_quad(
                key_, path_index_, static_cast<std::uint32_t>(next_group_++),
                domain_
            );
            component_index_ = 0U;
        }
        const auto component = component_index_++;
        if (component == 0U) return values_.first;
        if (component == 1U) return values_.second;
        if (component == 2U) return values_.third;
        return values_.fourth;
    }

private:
    PhiloxKey key_;
    std::uint64_t path_index_;
    std::uint32_t domain_;
    std::uint64_t next_group_ = 0ULL;
    RandomQuad values_{};
    std::uint32_t component_index_ = 4U;
};

// A model retains one path identity and step cursor, not a cached quad and
// Box--Muller pair for every possible source.
struct DomainRandomContext {
    PhiloxKey key;
    std::uint64_t path_index;
    std::uint32_t step_index = 0U;

    __device__ __forceinline__ DomainRandomContext(
        PhiloxKey row_key, std::uint64_t path
    ) : key(row_key), path_index(path) {}

    __device__ __forceinline__ std::uint32_t next_step() {
        if (step_index > kMaximumDomainStep) asm volatile("trap;");
        return step_index++;
    }

    template<std::uint8_t SourceId>
    __device__ __forceinline__ DomainUniformSequence source(
        std::uint32_t step
    ) const {
        return {key, path_index, source_step_domain<SourceId>(step)};
    }
};

// The CIR Poisson--Gamma mixture has two variable-consumption sources. Each
// transition gets fresh source/step addresses, so neither PTRS nor Gamma
// rejection can move the other law or a subsequent transition.
template<std::uint8_t PoissonSource, std::uint8_t GammaSource>
__device__ __forceinline__ float domain_scaled_noncentral_chi_square(
    DomainRandomContext& random,
    float degrees_of_freedom,
    float noncentrality,
    float scale
) {
    static_assert(PoissonSource != GammaSource);
    const auto step = random.next_step();
    auto poisson_uniforms = random.source<PoissonSource>(step);
    const auto poisson = poisson_from_uniform_sequence(
        poisson_uniforms, 0.5f * noncentrality
    );
    auto gamma_uniforms = random.source<GammaSource>(step);
    NormalPairCache gamma_cache;
    return marsaglia_tsang_gamma(
        gamma_uniforms, gamma_cache,
        0.5f * degrees_of_freedom + static_cast<float>(poisson),
        2.0f * scale
    );
}

static_assert(std::is_trivially_copyable_v<DomainUniformSequence>);
static_assert(std::is_trivially_copyable_v<DomainRandomContext>);

}  // namespace ai_factory::workbench::philox
