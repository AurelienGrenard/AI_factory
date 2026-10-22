// Host-resolved work for a gradient; price ownership is assigned by the kernel.
#pragma once
#include <cstdint>

namespace ai_factory::workbench::price_gradients {
enum class CentralRequirement : std::uint8_t {
    none = 0,       // Centered endpoints evolve independently of the central state.
    state = 1,      // Endpoints reuse the central trajectory, without its payoff.
    payoff = 2,     // One-sided stencil also consumes the central payoff per path.
};
}  // namespace ai_factory::workbench::price_gradients
