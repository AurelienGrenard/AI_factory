#pragma once

#include <cstddef>

namespace ai_factory::simulation {

struct TimeGrid {
    double maturity;
    std::size_t num_steps;
};

}  // namespace ai_factory::simulation
