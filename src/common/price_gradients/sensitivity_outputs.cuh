// Caller-owned price, gradient and diagonal-Hessian buffers.
#pragma once

#include "common/price_gradients/launch.cuh"

#include <cstddef>

namespace ai_factory::workbench::price_gradients {

struct SensitivityOutputs {
    float* prices;
    float* price_standard_errors;
    float* gradients;
    float* gradient_standard_errors;
    float* diagonal_hessians;
    float* diagonal_hessian_standard_errors;
    std::size_t price_capacity;
    std::size_t sensitivity_capacity;
};

inline SensitivityOutputs as_sensitivity_outputs(Outputs outputs) {
    return {
        outputs.prices,
        outputs.price_standard_errors,
        outputs.gradients,
        outputs.gradient_standard_errors,
        nullptr,
        nullptr,
        outputs.price_capacity,
        outputs.gradient_capacity,
    };
}

inline SensitivityOutputs as_sensitivity_outputs(
    SensitivityOutputs outputs
) {
    return outputs;
}

}  // namespace ai_factory::workbench::price_gradients
