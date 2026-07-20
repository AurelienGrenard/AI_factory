#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai_factory::simulation {

inline constexpr const char* kPhilox4x32_10BoxMuller =
    "philox4x32_10_box_muller";

struct Philox4x32Counter {
    std::uint32_t v0;
    std::uint32_t v1;
    std::uint32_t v2;
    std::uint32_t v3;
};

struct Philox4x32Key {
    std::uint32_t k0;
    std::uint32_t k1;
};

Philox4x32Counter philox4x32_10(
    Philox4x32Counter counter,
    Philox4x32Key key
);

std::vector<std::uint32_t> philox_uniform_uint32(
    std::uint64_t seed,
    std::size_t count,
    std::uint64_t stream = 0
);

std::vector<double> philox_uniforms(
    std::uint64_t seed,
    std::size_t count,
    std::uint64_t stream = 0
);

std::vector<double> philox_standard_normals(
    std::uint64_t seed,
    std::size_t count,
    std::uint64_t stream = 0
);

double philox_uniform(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t uniform_index
);

double philox_standard_normal(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t normal_index
);

class PhiloxUniformSequence {
public:
    PhiloxUniformSequence(
        std::uint64_t seed,
        std::uint64_t stream,
        std::uint64_t first_index = 0
    ) : seed_(seed), stream_(stream), index_(first_index) {}

    double next() {
        if (index_ / 4ULL != cached_block_) {
            refill();
        }
        return values_[index_++ % 4ULL];
    }

private:
    void refill();

    std::uint64_t seed_;
    std::uint64_t stream_;
    std::uint64_t index_;
    std::uint64_t cached_block_ = static_cast<std::uint64_t>(-1);
    double values_[4]{};
};

class PhiloxNormalSequence {
public:
    PhiloxNormalSequence(
        std::uint64_t seed,
        std::uint64_t stream,
        std::uint64_t first_index = 0
    ) : seed_(seed), stream_(stream), index_(first_index) {}

    double next() {
        if (index_ / 4ULL != cached_block_) {
            refill();
        }
        return values_[index_++ % 4ULL];
    }

private:
    void refill();

    std::uint64_t seed_;
    std::uint64_t stream_;
    std::uint64_t index_;
    std::uint64_t cached_block_ = static_cast<std::uint64_t>(-1);
    double values_[4]{};
};

}  // namespace ai_factory::simulation
