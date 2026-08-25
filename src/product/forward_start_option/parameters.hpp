// Forward-start-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct ForwardStartOptionParameters {
    float moneyness;
    std::uint32_t reset_time;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<ForwardStartOptionParameters>);

}  // namespace ai_factory::workbench::product
