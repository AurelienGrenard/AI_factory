// G2++ parameters shared by host loaders and curve-fitted analytics.
#pragma once

#include "model/fixed_income/g2/parameters.hpp"

#include <type_traits>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus {

// Curve-independent process parameters of one centered G2++ model.
struct ModelParameters {
    model::fixed_income::g2::ProcessParameters process;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus
