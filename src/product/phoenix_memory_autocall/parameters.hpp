// Phoenix-memory-autocall contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct PhoenixMemoryAutocallParameters {
    std::uint32_t maturity_days;
    std::uint32_t observation_interval_days;
    float autocall_barrier;
    float coupon_barrier;
    float protection_barrier;
    float annual_coupon_rate;
};

static_assert(std::is_trivially_copyable_v<PhoenixMemoryAutocallParameters>);

}  // namespace ai_factory::workbench::product
