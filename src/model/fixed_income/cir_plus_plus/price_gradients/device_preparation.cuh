// Device preparation adapter reusing the CIR factor parameter domain.
#pragma once

#include "model/fixed_income/cir/price_gradients/device_preparation.cuh"
#include "model/fixed_income/cir_plus_plus/parameters.hpp"

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::price_gradients {
using DevicePreparation = cir::price_gradients::DevicePreparation;
}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::price_gradients
