// Down-and-out-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct DownAndOutOptionParameters {
    float strike;
    float barrier;
    std::uint32_t maturity_days;
};

static_assert(std::is_trivially_copyable_v<DownAndOutOptionParameters>);

}  // namespace ai_factory::workbench::product
