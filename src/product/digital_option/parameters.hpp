// Digital-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct DigitalOptionParameters {
    float strike;
    std::uint32_t maturity_days;
    float cash_payoff;
};

static_assert(std::is_trivially_copyable_v<DigitalOptionParameters>);

}  // namespace ai_factory::workbench::product
