// Caller-owned buffers for selected mixed second derivatives.
#pragma once

#include <cstddef>

namespace ai_factory::workbench::price_gradients {

struct MixedSensitivityOutputs {
    float* hessians = nullptr;
    float* standard_errors = nullptr;
    std::size_t capacity = 0U;
};

}  // namespace ai_factory::workbench::price_gradients
