// Up-and-in-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct UpAndInOptionParameters {
    float strike;
    float barrier;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<UpAndInOptionParameters>);

}  // namespace ai_factory::workbench::product
