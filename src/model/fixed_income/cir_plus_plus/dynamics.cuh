// Public CIR++ dynamics contract: samples contain the unshifted CIR factor state y.
// Reuse the exact CIR transition and Philox consumption; r(t)=y(t)+phi(t).
#pragma once

#include "model/fixed_income/cir_plus_plus/parameters.hpp"
#include "model/fixed_income/cir/dynamics.cuh"

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus {
using DynamicsPolicy = cir::DynamicsPolicy;
}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus
