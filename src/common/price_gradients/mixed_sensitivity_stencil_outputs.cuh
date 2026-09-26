// Device output view for the row-local stencils used by mixed derivatives.
#pragma once

#include "common/price_gradients/mixed_sensitivity_stencil.cuh"

#include <cstddef>
#include <type_traits>

namespace ai_factory::workbench::price_gradients {

struct MixedSensitivityStencilOutputs {
    MixedSensitivityStencil* stencils = nullptr;
    std::size_t capacity = 0U;
};

static_assert(std::is_trivially_copyable_v<MixedSensitivityStencilOutputs>);

}  // namespace ai_factory::workbench::price_gradients
