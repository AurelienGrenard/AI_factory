// Provide compact compensated FP32 accumulation for path-local observables.
#pragma once

namespace ai_factory::workbench {

struct CompensatedFloatSum {
    float sum = 0.0f;
    float correction = 0.0f;

    __host__ __device__ __forceinline__ void add(float value) {
        const float corrected_value = value - correction;
        const float updated_sum = sum + corrected_value;
        correction = (updated_sum - sum) - corrected_value;
        sum = updated_sum;
    }

    __host__ __device__ __forceinline__ float value() const {
        return sum;
    }
};

}  // namespace ai_factory::workbench
