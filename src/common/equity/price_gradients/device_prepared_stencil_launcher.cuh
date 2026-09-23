// Equity compatibility name for the generic represented-stencil launcher.
#pragma once

#include "common/price_gradients/device_prepared_stencil_launcher.cuh"

namespace ai_factory::workbench::equity::price_gradients {

using ::ai_factory::workbench::price_gradients::
    prepare_device_sensitivity_stencils;

}  // namespace ai_factory::workbench::equity::price_gradients
