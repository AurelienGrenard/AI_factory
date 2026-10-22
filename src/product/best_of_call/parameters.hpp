// Best-of call on the maximum terminal spot relative to each initial spot.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct BestOfCallParameters {
    float strike; // dimensionless fraction of each asset's initial spot
    std::uint32_t maturity_days;
};

static_assert(std::is_trivially_copyable_v<BestOfCallParameters>);

}  // namespace ai_factory::workbench::product
