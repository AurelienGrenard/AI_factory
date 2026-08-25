// Cliquet contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct CliquetParameters {
    std::uint32_t maturity;
    std::uint32_t observation_interval;
    float participation_rate;
    float local_floor;
    float local_cap;
    float global_floor;
    float global_cap;
};

static_assert(std::is_trivially_copyable_v<CliquetParameters>);

}  // namespace ai_factory::workbench::product
