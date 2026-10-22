// Call on the arithmetic mean of initial-spot-normalized terminal assets.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct BasketCallParameters {
    float strike; // dimensionless fraction of the initial basket mean (1)
    std::uint32_t maturity_days;
};

static_assert(std::is_trivially_copyable_v<BasketCallParameters>);

}  // namespace ai_factory::workbench::product
