// Bates model parameters shared by host loaders and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::bates {

// Compact FP32 model row transferred from host memory to CUDA.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_variance;
    float kappa;
    float theta;
    float gamma;
    float rho;
    float jump_intensity;
    float jump_log_mean;
    float jump_log_volatility;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::bates
