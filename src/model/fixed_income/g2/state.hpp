// Path-local state shared by G2 parameters, dynamics, and analytics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::fixed_income::g2 {

struct State {
    float state_x;
    float state_y;
};

static_assert(std::is_trivially_copyable_v<State>);

}  // namespace ai_factory::workbench::model::fixed_income::g2
