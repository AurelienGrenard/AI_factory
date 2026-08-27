// FP64 continuation comparison shared by Longstaff-Schwartz update kernels.
#pragma once

#include <cuda_runtime.h>

namespace ai_factory::workbench::longstaff_schwartz {

__host__ __device__ __forceinline__ float select_exercise_cashflow(
    float immediate_value,
    double continuation_estimate,
    float continued_cashflow
) {
    return static_cast<double>(immediate_value) > continuation_estimate
        ? immediate_value
        : continued_cashflow;
}

}  // namespace ai_factory::workbench::longstaff_schwartz
