// Phoenix-autocall contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct PhoenixAutocallParameters {
    std::uint32_t maturity;
    std::uint32_t observation_interval;
    float autocall_barrier;
    float coupon_barrier;
    float protection_barrier;
    float annual_coupon_rate;
};

static_assert(std::is_trivially_copyable_v<PhoenixAutocallParameters>);

}  // namespace ai_factory::workbench::product
