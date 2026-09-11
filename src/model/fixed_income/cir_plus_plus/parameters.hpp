// CIR++ parameters describe its nonnegative CIR factor, not the shifted rate.
// The independent initial curve is supplied only to fitted analytics/pricing.
#pragma once

#include "model/fixed_income/cir/parameters.hpp"

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus {
using ProcessParameters = cir::ProcessParameters;
using ModelParameters = cir::ModelParameters;
}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus
