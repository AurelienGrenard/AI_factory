// Worst-of call on the minimum terminal spot relative to each initial spot.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct WorstOfCallParameters {
    float strike; // dimensionless fraction of initial spot
    std::uint32_t maturity_days;
};

static_assert(std::is_trivially_copyable_v<WorstOfCallParameters>);

}  // namespace ai_factory::workbench::product
