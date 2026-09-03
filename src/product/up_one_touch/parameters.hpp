// Up-one-touch contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct UpOneTouchParameters {
    float barrier;
    float cash_payoff;
    std::uint32_t maturity_days;
};

static_assert(std::is_trivially_copyable_v<UpOneTouchParameters>);

}  // namespace ai_factory::workbench::product
