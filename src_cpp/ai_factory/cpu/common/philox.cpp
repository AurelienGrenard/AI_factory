#include "ai_factory/cpu/common/philox.hpp"

#include <cmath>

namespace ai_factory::simulation {
namespace {

constexpr std::uint32_t kM0 = 0xD2511F53U;
constexpr std::uint32_t kM1 = 0xCD9E8D57U;
constexpr std::uint32_t kW0 = 0x9E3779B9U;
constexpr std::uint32_t kW1 = 0xBB67AE85U;
constexpr double kUInt32Scale = 1.0 / 4294967296.0;
constexpr double kPi = 3.141592653589793238462643383279502884;

Philox4x32Key seed_to_key(std::uint64_t seed) {
    return {
        static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL),
        static_cast<std::uint32_t>(seed >> 32U),
    };
}

}  // namespace

Philox4x32Counter philox4x32_10(
    Philox4x32Counter counter,
    Philox4x32Key key
) {
    auto ctr = counter;
    auto current_key = key;

    for (int round = 0; round < 10; ++round) {
        const std::uint64_t product0 =
            static_cast<std::uint64_t>(kM0) * ctr.v0;
        const std::uint64_t product1 =
            static_cast<std::uint64_t>(kM1) * ctr.v2;
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

std::vector<std::uint32_t> philox_uniform_uint32(
    std::uint64_t seed,
    std::size_t count,
    std::uint64_t stream
) {
    std::vector<std::uint32_t> values(count);
    const auto key = seed_to_key(seed);
    const std::size_t blocks = (count + 3U) / 4U;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(blocks >= 1024U)
#endif
    for (std::ptrdiff_t signed_block = 0;
         signed_block < static_cast<std::ptrdiff_t>(blocks);
         ++signed_block) {
        const auto block = static_cast<std::size_t>(signed_block);
        const Philox4x32Counter counter{
            static_cast<std::uint32_t>(block & 0xFFFFFFFFULL),
            static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(block) >> 32U) & 0xFFFFFFFFULL
            ),
            static_cast<std::uint32_t>(stream & 0xFFFFFFFFULL),
            static_cast<std::uint32_t>((stream >> 32U) & 0xFFFFFFFFULL),
        };
        const auto output = philox4x32_10(counter, key);
        const auto offset = 4U * block;
        const std::uint32_t generated[4] = {
            output.v0, output.v1, output.v2, output.v3
        };
        for (std::size_t lane = 0; lane < 4U && offset + lane < count; ++lane) {
            values[offset + lane] = generated[lane];
        }
    }

    return values;
}

std::vector<double> philox_uniforms(
    std::uint64_t seed,
    std::size_t count,
    std::uint64_t stream
) {
    const auto raw = philox_uniform_uint32(seed, count, stream);
    std::vector<double> uniforms(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096U)
#endif
    for (std::ptrdiff_t signed_index = 0;
         signed_index < static_cast<std::ptrdiff_t>(count);
         ++signed_index) {
        const auto index = static_cast<std::size_t>(signed_index);
        uniforms[index] =
            (static_cast<double>(raw[index]) + 0.5) * kUInt32Scale;
    }
    return uniforms;
}

std::vector<double> philox_standard_normals(
    std::uint64_t seed,
    std::size_t count,
    std::uint64_t stream
) {
    const std::size_t pairs = (count + 1U) / 2U;
    const auto uniforms = philox_uniforms(seed, 2U * pairs, stream);
    std::vector<double> normals(2U * pairs);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(pairs >= 2048U)
#endif
    for (std::ptrdiff_t signed_pair = 0;
         signed_pair < static_cast<std::ptrdiff_t>(pairs);
         ++signed_pair) {
        const auto pair = static_cast<std::size_t>(signed_pair);
        const double u1 = uniforms[2U * pair];
        const double u2 = uniforms[2U * pair + 1U];
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = 2.0 * kPi * u2;
        normals[2U * pair] = radius * std::cos(angle);
        normals[2U * pair + 1U] = radius * std::sin(angle);
    }

    normals.resize(count);
    return normals;
}

double philox_uniform(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t uniform_index
) {
    const auto block = uniform_index / 4ULL;
    const Philox4x32Counter counter{
        static_cast<std::uint32_t>(block),
        static_cast<std::uint32_t>(block >> 32U),
        static_cast<std::uint32_t>(stream),
        static_cast<std::uint32_t>(stream >> 32U),
    };
    const auto output = philox4x32_10(counter, seed_to_key(seed));
    const std::uint32_t values[4] = {output.v0, output.v1, output.v2, output.v3};
    return (static_cast<double>(values[uniform_index % 4ULL]) + 0.5)
           * kUInt32Scale;
}

double philox_standard_normal(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t normal_index
) {
    const auto pair = normal_index / 2ULL;
    const double u1 = philox_uniform(seed, stream, 2ULL * pair);
    const double u2 = philox_uniform(seed, stream, 2ULL * pair + 1ULL);
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = 2.0 * std::acos(-1.0) * u2;
    return normal_index % 2ULL == 0ULL ? radius * std::cos(angle)
                                       : radius * std::sin(angle);
}

void PhiloxUniformSequence::refill() {
    const auto block = index_ / 4ULL;
    const auto output = philox4x32_10(
        {
            static_cast<std::uint32_t>(block),
            static_cast<std::uint32_t>(block >> 32U),
            static_cast<std::uint32_t>(stream_),
            static_cast<std::uint32_t>(stream_ >> 32U),
        },
        seed_to_key(seed_)
    );
    const std::uint32_t raw[4] = {
        output.v0, output.v1, output.v2, output.v3
    };
    for (int lane = 0; lane < 4; ++lane) {
        values_[lane] = (static_cast<double>(raw[lane]) + 0.5) * kUInt32Scale;
    }
    cached_block_ = block;
}

void PhiloxNormalSequence::refill() {
    const auto block = index_ / 4ULL;
    const auto output = philox4x32_10(
        {
            static_cast<std::uint32_t>(block),
            static_cast<std::uint32_t>(block >> 32U),
            static_cast<std::uint32_t>(stream_),
            static_cast<std::uint32_t>(stream_ >> 32U),
        },
        seed_to_key(seed_)
    );
    const double uniforms[4] = {
        (static_cast<double>(output.v0) + 0.5) * kUInt32Scale,
        (static_cast<double>(output.v1) + 0.5) * kUInt32Scale,
        (static_cast<double>(output.v2) + 0.5) * kUInt32Scale,
        (static_cast<double>(output.v3) + 0.5) * kUInt32Scale,
    };
    for (int pair = 0; pair < 2; ++pair) {
        const double radius = std::sqrt(-2.0 * std::log(uniforms[2 * pair]));
        const double angle = 2.0 * kPi * uniforms[2 * pair + 1];
        values_[2 * pair] = radius * std::cos(angle);
        values_[2 * pair + 1] = radius * std::sin(angle);
    }
    cached_block_ = block;
}

}  // namespace ai_factory::simulation
