// Host-side Philox-4x32-10 used to materialize reproducible model rows.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::offline::sampling {

inline constexpr std::uint32_t kHostPhiloxM0 = 0xD2511F53U;
inline constexpr std::uint32_t kHostPhiloxM1 = 0xCD9E8D57U;
inline constexpr std::uint32_t kHostPhiloxW0 = 0x9E3779B9U;
inline constexpr std::uint32_t kHostPhiloxW1 = 0xBB67AE85U;
inline constexpr float kHostLargestUniform = 0x1.fffffep-1f;
inline constexpr std::uint64_t kHostParameterSamplingDomain =
    0x6a09e667f3bcc909ULL;

struct HostPhiloxCounter {
    std::uint32_t v0;
    std::uint32_t v1;
    std::uint32_t v2;
    std::uint32_t v3;
};

struct HostPhiloxKey {
    std::uint32_t k0;
    std::uint32_t k1;
};

inline HostPhiloxKey host_philox_key(std::uint64_t seed) noexcept {
    return {
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32U),
    };
}

inline HostPhiloxCounter host_philox4x32_10(
    HostPhiloxKey key,
    HostPhiloxCounter counter
) noexcept {
    for (int round = 0; round < 10; ++round) {
        const std::uint64_t product0 =
            static_cast<std::uint64_t>(kHostPhiloxM0) * counter.v0;
        const std::uint64_t product1 =
            static_cast<std::uint64_t>(kHostPhiloxM1) * counter.v2;
        const std::uint32_t hi0 = static_cast<std::uint32_t>(product0 >> 32U);
        const std::uint32_t hi1 = static_cast<std::uint32_t>(product1 >> 32U);
        counter = {
            static_cast<std::uint32_t>(hi1 ^ counter.v1 ^ key.k0),
            static_cast<std::uint32_t>(product1),
            static_cast<std::uint32_t>(hi0 ^ counter.v3 ^ key.k1),
            static_cast<std::uint32_t>(product0),
        };
        if (round != 9) {
            key.k0 += kHostPhiloxW0;
            key.k1 += kHostPhiloxW1;
        }
    }
    return counter;
}

class HostUniformSequence {
public:
    HostUniformSequence(std::uint64_t seed, std::uint64_t row_index)
        : key_(host_philox_key(seed ^ kHostParameterSamplingDomain)),
          row_index_(row_index) {
        refill();
    }

    float next() noexcept {
        if (component_ == 4U) refill();
        const std::uint32_t word = component_ == 0U ? values_.v0
            : component_ == 1U ? values_.v1
            : component_ == 2U ? values_.v2
            : values_.v3;
        ++component_;
        const double midpoint =
            (static_cast<double>(word) + 0.5) * 0x1p-32;
        return std::min(
            static_cast<float>(midpoint),
            kHostLargestUniform
        );
    }

private:
    void refill() noexcept {
        values_ = host_philox4x32_10(
            key_,
            {
                static_cast<std::uint32_t>(row_index_),
                static_cast<std::uint32_t>(row_index_ >> 32U),
                static_cast<std::uint32_t>(group_index_),
                static_cast<std::uint32_t>(group_index_ >> 32U),
            }
        );
        ++group_index_;
        component_ = 0U;
    }

    HostPhiloxKey key_;
    std::uint64_t row_index_;
    std::uint64_t group_index_ = 0U;
    HostPhiloxCounter values_{};
    std::uint32_t component_ = 4U;
};

struct HostUniformBounds {
    float minimum;
    float maximum;
};

inline float uniform(
    HostUniformBounds bounds,
    HostUniformSequence& uniforms
) noexcept {
    return std::fma(
        bounds.maximum - bounds.minimum,
        uniforms.next(),
        bounds.minimum
    );
}

}  // namespace ai_factory::workbench::offline::sampling
