// Compact ownership and index of one host-resolved sensitivity parameter.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::price_gradients {

enum class SensitivityParameterOwner : std::uint8_t {
    model,
    curve,
    product,
    maturity,
};

struct SensitivityParameter {
    std::uint8_t code;
};

static_assert(std::is_trivially_copyable_v<SensitivityParameter>);
static_assert(sizeof(SensitivityParameter) == 1U);

}  // namespace ai_factory::workbench::price_gradients
