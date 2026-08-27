// Asian-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct AsianOptionParameters {
    float strike;
    std::uint32_t maturity_days;
};

static_assert(std::is_trivially_copyable_v<AsianOptionParameters>);

}  // namespace ai_factory::workbench::product
