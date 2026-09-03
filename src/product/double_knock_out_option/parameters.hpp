// Double-knock-out-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct DoubleKnockOutOptionParameters {
    float strike;
    float lower_barrier;
    float upper_barrier;
    std::uint32_t maturity_days;
};

static_assert(std::is_trivially_copyable_v<DoubleKnockOutOptionParameters>);

}  // namespace ai_factory::workbench::product
