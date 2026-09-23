// Calendar representation shared by gradient recipes and equity scenario plans.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::price_gradients {

struct TimeConfiguration {
    float dt = 1.0f / 504.0f;
    std::uint32_t simulation_steps_per_day = 2U;

    void validate() const {
        if (!std::isfinite(dt) || !(dt > 0.0f)
            || simulation_steps_per_day == 0U) {
            throw std::invalid_argument("Invalid gradient time grid.");
        }
    }
};

}  // namespace ai_factory::workbench::price_gradients
