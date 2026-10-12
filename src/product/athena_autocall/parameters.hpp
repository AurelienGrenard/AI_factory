// Athena-autocall contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct AthenaAutocallParameters {
    std::uint32_t maturity_days;
    std::uint32_t observation_interval_days;
    float autocall_barrier;
    float protection_barrier;
    float annual_coupon_rate;
    float reference_spot; // Frozen S0 for unit-notional loss redemption.
};

static_assert(std::is_trivially_copyable_v<AthenaAutocallParameters>);

}  // namespace ai_factory::workbench::product
