#pragma once

#include <cstdint>

namespace ai_factory::cuda {
namespace rng {

constexpr std::uint32_t kM0 = 0xD2511F53U;
constexpr std::uint32_t kM1 = 0xCD9E8D57U;
constexpr std::uint32_t kW0 = 0x9E3779B9U;
constexpr std::uint32_t kW1 = 0xBB67AE85U;
constexpr double kUInt32Scale = 1.0 / 4294967296.0;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct PhiloxCounter {
    std::uint32_t v0;
    std::uint32_t v1;
    std::uint32_t v2;
    std::uint32_t v3;
};

struct PhiloxKey {
    std::uint32_t k0;
    std::uint32_t k1;
};

struct NormalPair {
    double first;
    double second;
};

struct RandomQuad {
    double first;
    double second;
    double third;
    double fourth;
};

__device__ __forceinline__ PhiloxCounter philox4x32_10(
    PhiloxCounter counter,
    PhiloxKey key
) {
    auto ctr = counter;
    auto current_key = key;
    for (int round = 0; round < 10; ++round) {
        const auto product0 = static_cast<unsigned long long>(kM0) * ctr.v0;
        const auto product1 = static_cast<unsigned long long>(kM1) * ctr.v2;
        const auto hi0 = static_cast<std::uint32_t>(product0 >> 32U);
        const auto hi1 = static_cast<std::uint32_t>(product1 >> 32U);
        const auto lo0 = static_cast<std::uint32_t>(product0);
        const auto lo1 = static_cast<std::uint32_t>(product1);
        ctr = {
            static_cast<std::uint32_t>(hi1 ^ ctr.v1 ^ current_key.k0),
            lo1,
            static_cast<std::uint32_t>(hi0 ^ ctr.v3 ^ current_key.k1),
            lo0,
        };
        if (round != 9) {
            current_key.k0 += kW0;
            current_key.k1 += kW1;
        }
    }
    return ctr;
}

__device__ __forceinline__ double uint32_to_uniform(std::uint32_t value) {
    return (static_cast<double>(value) + 0.5) * kUInt32Scale;
}

__device__ __forceinline__ double standard_uniform(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t uniform_index
) {
    const auto block_index = uniform_index / 4ULL;
    const PhiloxCounter counter{
        static_cast<std::uint32_t>(block_index & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((block_index >> 32U) & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>(stream & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((stream >> 32U) & 0xFFFFFFFFULL),
    };
    const PhiloxKey key{
        static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((seed >> 32U) & 0xFFFFFFFFULL),
    };
    const auto output = philox4x32_10(counter, key);
    const auto component = uniform_index % 4ULL;
    if (component == 0ULL) {
        return uint32_to_uniform(output.v0);
    }
    if (component == 1ULL) {
        return uint32_to_uniform(output.v1);
    }
    if (component == 2ULL) {
        return uint32_to_uniform(output.v2);
    }
    return uint32_to_uniform(output.v3);
}

__device__ __forceinline__ double standard_normal(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t normal_index
) {
    const auto pair_index = normal_index / 2ULL;
    const auto block_index = pair_index / 2ULL;
    const PhiloxCounter counter{
        static_cast<std::uint32_t>(block_index & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((block_index >> 32U) & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>(stream & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((stream >> 32U) & 0xFFFFFFFFULL),
    };
    const PhiloxKey key{
        static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((seed >> 32U) & 0xFFFFFFFFULL),
    };
    const auto output = philox4x32_10(counter, key);
    const bool use_high_pair = (pair_index % 2ULL) == 1ULL;
    const double u1 = uint32_to_uniform(use_high_pair ? output.v2 : output.v0);
    const double u2 = uint32_to_uniform(use_high_pair ? output.v3 : output.v1);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 2.0 * kPi * u2;
    return (normal_index % 2ULL == 0ULL) ? radius * cos(angle)
                                         : radius * sin(angle);
}

__device__ __forceinline__ NormalPair standard_normal_pair(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t pair_index
) {
    const auto block_index = pair_index / 2ULL;
    const PhiloxCounter counter{
        static_cast<std::uint32_t>(block_index & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((block_index >> 32U) & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>(stream & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((stream >> 32U) & 0xFFFFFFFFULL),
    };
    const PhiloxKey key{
        static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((seed >> 32U) & 0xFFFFFFFFULL),
    };
    const auto output = philox4x32_10(counter, key);
    const bool use_high_pair = (pair_index % 2ULL) == 1ULL;
    const double u1 = uint32_to_uniform(use_high_pair ? output.v2 : output.v0);
    const double u2 = uint32_to_uniform(use_high_pair ? output.v3 : output.v1);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 2.0 * kPi * u2;
    double sine = 0.0;
    double cosine = 0.0;
    sincos(angle, &sine, &cosine);
    return {radius * cosine, radius * sine};
}

__device__ __forceinline__ RandomQuad standard_uniform_quad(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t block_index
) {
    const PhiloxCounter counter{
        static_cast<std::uint32_t>(block_index & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((block_index >> 32U) & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>(stream & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((stream >> 32U) & 0xFFFFFFFFULL),
    };
    const PhiloxKey key{
        static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>((seed >> 32U) & 0xFFFFFFFFULL),
    };
    const auto output = philox4x32_10(counter, key);
    return {
        uint32_to_uniform(output.v0),
        uint32_to_uniform(output.v1),
        uint32_to_uniform(output.v2),
        uint32_to_uniform(output.v3),
    };
}

__device__ __forceinline__ RandomQuad standard_normal_quad(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t block_index
) {
    const auto uniforms = standard_uniform_quad(seed, stream, block_index);
    const double first_radius = sqrt(-2.0 * log(uniforms.first));
    const double first_angle = 2.0 * kPi * uniforms.second;
    const double second_radius = sqrt(-2.0 * log(uniforms.third));
    const double second_angle = 2.0 * kPi * uniforms.fourth;
    double first_sine = 0.0;
    double first_cosine = 0.0;
    double second_sine = 0.0;
    double second_cosine = 0.0;
    sincos(first_angle, &first_sine, &first_cosine);
    sincos(second_angle, &second_sine, &second_cosine);
    return {
        first_radius * first_cosine,
        first_radius * first_sine,
        second_radius * second_cosine,
        second_radius * second_sine,
    };
}

class NormalSequence {
public:
    __device__ __forceinline__ NormalSequence(
        std::uint64_t seed,
        std::uint64_t stream,
        std::uint64_t first_index
    ) : seed_(seed), stream_(stream), index_(first_index) {}

    __device__ __forceinline__ double next() {
        const auto block_index = index_ / 4ULL;
        if (!has_block_ || block_index != cached_block_index_) {
            values_ = standard_normal_quad(seed_, stream_, block_index);
            cached_block_index_ = block_index;
            has_block_ = true;
        }
        const auto component = index_++ % 4ULL;
        if (component == 0ULL) {
            return values_.first;
        }
        if (component == 1ULL) {
            return values_.second;
        }
        if (component == 2ULL) {
            return values_.third;
        }
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

class UniformSequence {
public:
    __device__ __forceinline__ UniformSequence(
        std::uint64_t seed,
        std::uint64_t stream,
        std::uint64_t first_index
    ) : seed_(seed), stream_(stream), index_(first_index) {}

    __device__ __forceinline__ double next() {
        const auto block_index = index_ / 4ULL;
        if (!has_block_ || block_index != cached_block_index_) {
            values_ = standard_uniform_quad(seed_, stream_, block_index);
            cached_block_index_ = block_index;
            has_block_ = true;
        }
        const auto component = index_++ % 4ULL;
        if (component == 0ULL) {
            return values_.first;
        }
        if (component == 1ULL) {
            return values_.second;
        }
        if (component == 2ULL) {
            return values_.third;
        }
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

}  // namespace rng
}  // namespace ai_factory::cuda
